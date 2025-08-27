import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { LanguageClient, LanguageClientOptions, ServerOptions, Executable } from 'vscode-languageclient/node';
import * as https from 'https';
import { spawn } from 'child_process';

const TERMINAL_NAME = 'JMCC Terminal';
const ASSETS_DIR_NAME = 'assets';
const JMCC_DIR_NAME = 'JMCC';
const COMPILER_SCRIPT_NAME = 'jmcc.py';
const PROPS_FILE_NAME = 'jmcc.properties';
const COMPLETIONS_FILE_NAME = 'completions.json';
const CONFIG_FILE_NAME = '.jmccconfig.json';
const OBFUSCATION_MAPPING_EXTENSION = '.obfmap';

const REMOTE_PROPS_URL = 'https://raw.githubusercontent.com/donzgold/JustMC_compilator/master/jmcc.properties';
const COMPLETIONS_URL = 'https://raw.githubusercontent.com/donzgold/JustMC_compilator/master/assets/completions.json';
const JMCC_PY_URL = 'https://raw.githubusercontent.com/donzgold/JustMC_compilator/master/jmcc.py';

interface JMCCConfig {
  compilerPath: string;
  compilerOutputPath: string;
}

interface JsonFunctionCall {
  type: 'text';
  text: string;
  value?: any;
}

interface JsonValue {
  type?: string;
  variable?: string;
  scope?: string;
  text?: string;
  name?: string;
  parsing?: string;
  [key: string]: any;
}

interface NameMapping {
  [key: string]: string;
}

let client: LanguageClient | undefined;

function getSettings(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration('jmcc-helper');
}

async function ensureDocumentSaved(uri: vscode.Uri): Promise<void> {
  const doc = await vscode.workspace.openTextDocument(uri);
  if (doc.isDirty) {
    await doc.save();
  }
}

function shouldClearTerminal(): boolean {
  return getSettings().get<boolean>('clearTerminalBeforeCommand', true);
}

function getDefaultCompileMode(): 'UPLOAD' | 'SAVE' | 'BOTH' | 'OBFUSCATE SAVE' | 'OBFUSCATE URL' {
  return getSettings()
    .get<string>('defaultCompileActiveFileMode', 'UPLOAD')
    .toUpperCase() as 'UPLOAD' | 'SAVE' | 'BOTH' | 'OBFUSCATE SAVE' | 'OBFUSCATE URL';
}

function getWorkspaceFolder(fileUri: vscode.Uri): vscode.WorkspaceFolder | undefined {
  return vscode.workspace.getWorkspaceFolder(fileUri);
}

function getPythonCmd(): string {
  return os.platform() === 'win32' ? 'py' : 'python3';
}

function getPaths(context: vscode.ExtensionContext) {
  const assetsDir = context.asAbsolutePath(path.join('out', ASSETS_DIR_NAME));
  const jmccDir = context.asAbsolutePath(path.join('out', JMCC_DIR_NAME));
  return { assetsDir, jmccDir };
}

function getConfigPath(workspaceFolder: vscode.WorkspaceFolder): string {
  const configDir = path.join(workspaceFolder.uri.fsPath, '.vscode');
  return path.join(configDir, CONFIG_FILE_NAME);
}

function loadOrInitConfig(workspaceFolder: vscode.WorkspaceFolder): JMCCConfig {
  const configPath = getConfigPath(workspaceFolder);
  const configDir = path.dirname(configPath);
  const autoCreateConfig = getSettings().get<boolean>('autoCreateConfig', true);

  const defaultConfig: JMCCConfig = {
    compilerPath: '',
    compilerOutputPath: ''
  };

  if (!fs.existsSync(configPath)) {
    if (autoCreateConfig) {
      if (!fs.existsSync(configDir)) fs.mkdirSync(configDir, { recursive: true });
      fs.writeFileSync(configPath, JSON.stringify(defaultConfig, null, 2), 'utf-8');
    }
    return defaultConfig;
  }

  let config: JMCCConfig;
  try {
    config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  } catch (err) {
    vscode.window.showErrorMessage(`JMCC: Ошибка чтения ${path.basename(configPath)}. Проверьте формат JSON.`);
    throw err;
  }

  let modified = false;
  if (typeof config.compilerPath !== 'string') { config.compilerPath = ''; modified = true; }
  if (typeof config.compilerOutputPath !== 'string') { config.compilerOutputPath = ''; modified = true; }
  if (modified && autoCreateConfig) {
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');
  }
  return config;
}

function validatePath(filePath: string, isFile: boolean = false): boolean {
  try {
    if (!fs.existsSync(filePath)) return false;
    const stat = fs.statSync(filePath);
    if (isFile && stat.isDirectory()) return false;
    return true;
  } catch (err) {
    console.error(`Ошибка проверки пути ${filePath}:`, err);
    return false;
  }
}

function getCompilerPath(context: vscode.ExtensionContext, config: JMCCConfig): string {
  const settingPath = getSettings().get<string>('compilerPath', '').trim();

  if (config.compilerPath?.trim()) return path.normalize(config.compilerPath);
  if (settingPath) return path.normalize(settingPath);

  const { jmccDir } = getPaths(context);
  return path.join(jmccDir, COMPILER_SCRIPT_NAME);
}

function validateCompilerPath(compilerPath: string, _context: vscode.ExtensionContext, config: JMCCConfig): boolean {
  if (config.compilerPath?.trim()) {
    if (!validatePath(compilerPath)) {
      vscode.window.showErrorMessage(`JMCC: Путь к компилятору недействителен: ${compilerPath}`);
      return false;
    }
  }
  return true;
}

function getOrCreateTerminal(): vscode.Terminal {
  const existing = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
  return existing ?? vscode.window.createTerminal(TERMINAL_NAME);
}

function runCommand(command: string, _workspaceFolder: vscode.WorkspaceFolder) {
  const terminal = getOrCreateTerminal();
  terminal.show();

  if (shouldClearTerminal()) {
    const clearCmd = os.platform() === 'win32' ? 'cls' : 'clear';
    terminal.sendText(clearCmd);
  }
  terminal.sendText(command);
}

function buildBaseCommandArgs(compilerPath: string, targetPath: string): string[] {
  const pythonCommand = getPythonCmd();
  return [pythonCommand, `"${compilerPath}"`, 'compile', `"${path.normalize(targetPath)}"`];
}

async function executeCompilationArgs(args: string[], cwd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(getPythonCmd(), args, { cwd, stdio: 'pipe', shell: false });
    let stdout = '';
    let stderr = '';
    child.stdout?.on('data', d => stdout += d.toString());
    child.stderr?.on('data', d => stderr += d.toString());
    child.on('close', code => {
      if (code !== 0) {
        reject(new Error(`Compilation failed with code ${code}: ${stderr}`));
      } else {
        resolve(stdout);
      }
    });
    child.on('error', err => reject(new Error(`Spawn error: ${err.message}`)));
  });
}

async function download(url: string, maxRedirects = 5, timeoutMs = 10000): Promise<string | null> {
  return new Promise((resolve) => {
    const visited = new Set<string>();
    function fetchOnce(currentUrl: string, redirectsLeft: number) {
      if (visited.has(currentUrl)) return resolve(null);
      visited.add(currentUrl);

      const req = https.get(currentUrl, res => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          if (redirectsLeft <= 0) return resolve(null);
          res.resume();
          const nextUrl = new URL(res.headers.location, currentUrl).toString();
          return fetchOnce(nextUrl, redirectsLeft - 1);
        }

        if (res.statusCode !== 200) {
          res.resume();
          return resolve(null);
        }

        let data = '';
        res.setEncoding('utf8');
        res.on('data', chunk => data += chunk);
        res.on('end', () => resolve(data));
      });

      req.setTimeout(timeoutMs, () => {
        req.destroy(new Error('Request timeout'));
      });

      req.on('error', () => resolve(null));
    }

    fetchOnce(url.trim(), maxRedirects);
  });
}

function extractDataVersion(content: string): string | null {
  const match = content.match(/^\s*data_version\s*=\s*(\S+)/m);
  return match ? match[1].trim() : null;
}

function upsertProp(lines: string[], key: string, value: string): string[] {
  const re = new RegExp(`^\\s*(#\\s*)?${key}\\s*=`, 'i');
  let found = false;
  const updated = lines.map(line => {
    if (re.test(line)) {
      found = true;
      return `${key} = ${value}`;
    }
    return line;
  });
  if (!found) updated.push(`${key} = ${value}`);
  return updated;
}

async function waitForFile(filePath: string, timeoutMs: number = 5000): Promise<boolean> {
  const start = Date.now();
  return new Promise(resolve => {
    const check = () => {
      if (fs.existsSync(filePath)) return resolve(true);
      if (Date.now() - start > timeoutMs) return resolve(false);
      setTimeout(check, 100);
    };
    check();
  });
}

function compile(targetPath: string, mode: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext, isCompileAsFile: boolean = false) {
  const config = loadOrInitConfig(workspaceFolder);
  const compilerPath = getCompilerPath(context, config);

  if (!validateCompilerPath(compilerPath, context, config)) return;

  const args = buildBaseCommandArgs(compilerPath, targetPath);

  if (mode === 'UPLOAD') args.push('-u');
  else if (mode === 'BOTH') args.push('-su');

  const settingOut = getSettings().get<string>('compilerOutputPath', '').trim();
  const outPath = (config.compilerOutputPath?.trim() || settingOut) ?? '';

  if ((mode === 'SAVE' || mode === 'BOTH' || isCompileAsFile) && outPath) {
    const normalizedOutputPath = path.normalize(outPath);
    if (!validatePath(normalizedOutputPath, false)) {
      vscode.window.showErrorMessage(`JMCC: Неверный путь для вывода: ${normalizedOutputPath}`);
      return;
    }
    args.push('-o', `"${normalizedOutputPath}"`);
  }

  runCommand(args.join(' '), workspaceFolder);
}

function decompile(targetPath: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext) {
  const config = loadOrInitConfig(workspaceFolder);
  const compilerPath = getCompilerPath(context, config);

  if (!validateCompilerPath(compilerPath, context, config)) return;

  const pythonCommand = getPythonCmd();
  const args = [pythonCommand, `"${compilerPath}"`, 'decompile', `"${path.normalize(targetPath)}"`];
  runCommand(args.join(' '), workspaceFolder);
}

function saveAndUpload(targetPath: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext) {
  const config = loadOrInitConfig(workspaceFolder);
  const compilerPath = getCompilerPath(context, config);

  if (!validateCompilerPath(compilerPath, context, config)) return;

  const pythonCommand = getPythonCmd();
  const args = [pythonCommand, `"${compilerPath}"`, 'compile', `"${path.normalize(targetPath)}"`, '-su'];

  if (config.compilerOutputPath?.trim()) {
    const normalizedOutputPath = path.normalize(config.compilerOutputPath);
    if (!validatePath(normalizedOutputPath, true)) {
      vscode.window.showErrorMessage(`JMCC: Путь для вывода должен быть файлом: ${normalizedOutputPath}`);
      return;
    }
    args.push('-o', `"${normalizedOutputPath}"`);
  }

  runCommand(args.join(' '), workspaceFolder);
}

async function compileWithObfuscation(filePath: string, mode: 'FILE' | 'URL', workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext) {
  const config = loadOrInitConfig(workspaceFolder);
  const compilerPath = getCompilerPath(context, config);
  const outputPath = filePath + '.json';
  const decompiledPath = outputPath.replace(/\.json$/, '.jc');

  try {
    if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath);
    if (fs.existsSync(decompiledPath)) fs.unlinkSync(decompiledPath);
    await executeCompilationArgs([compilerPath, 'compile', filePath], workspaceFolder.uri.fsPath);

    let attempts = 0;
    while (!fs.existsSync(outputPath) && attempts < 50) {
      await new Promise(r => setTimeout(r, 100));
      attempts++;
    }
    if (!fs.existsSync(outputPath)) throw new Error(`JSON file not found after compilation: ${outputPath}`);
    const jsonContent = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
    const obfuscator = new Obfuscator();
    const obfuscatedJson = obfuscator.obfuscateJson(jsonContent);
    fs.writeFileSync(outputPath, JSON.stringify(obfuscatedJson, null, 2));
    obfuscator.saveMapping(outputPath);

    if (mode === 'URL') {
      await executeCompilationArgs([compilerPath, 'decompile', outputPath], workspaceFolder.uri.fsPath);
      attempts = 0;
      while (!fs.existsSync(decompiledPath) && attempts < 50) {
        await new Promise(r => setTimeout(r, 100));
        attempts++;
      }
      if (!fs.existsSync(decompiledPath)) throw new Error(`Decompiled file not found: ${decompiledPath}`);
      const terminal = getOrCreateTerminal();
      const finalCompileCmd = `${getPythonCmd()} "${compilerPath}" compile "${decompiledPath}" -u`;
      let cleanupCmd: string;

      if (process.platform === 'win32') {
        const escapedJson = outputPath.replace(/"/g, '`"');
        const escapedJc = decompiledPath.replace(/"/g, '`"');
        cleanupCmd = `${finalCompileCmd}; if ($?) { del "${escapedJson}"; del "${escapedJc}" }`;
      } else {
        const qJson = outputPath.includes(' ') ? `"${outputPath}"` : outputPath;
        const qJc = decompiledPath.includes(' ') ? `"${decompiledPath}"` : decompiledPath;
        cleanupCmd = `${finalCompileCmd} && rm ${qJson} ${qJc}`;
      }

      terminal.show();
      terminal.sendText(cleanupCmd, true);
    }

    vscode.window.showInformationMessage(`Obfuscation and compilation completed successfully for ${filePath}`);
  } catch (error: any) {
    vscode.window.showErrorMessage(`Compilation/Obfuscation failed: ${error?.message || 'Unknown error'}`);
    console.error(error);
  }
}

async function decompileWithObfmap(filePath: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext) {
  const obfmapPath = filePath.replace(/\.json$/, OBFUSCATION_MAPPING_EXTENSION);
  if (!fs.existsSync(obfmapPath)) {
    vscode.window.showErrorMessage('Obfuscation mapping file (.obfmap) not found');
    return;
  }

  try {
    const jsonContent = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const obfuscator = new Obfuscator();
    obfuscator.loadMapping(obfmapPath);
    const deobfuscatedJson = obfuscator.obfuscateJson(jsonContent);
    fs.writeFileSync(filePath, JSON.stringify(deobfuscatedJson, null, 2));
    decompile(filePath, workspaceFolder, context);
  } catch (error: any) {
    vscode.window.showErrorMessage(`Error during deobfuscation: ${error?.message || String(error)}`);
  }
}

class Obfuscator {
  private nameMapping: NameMapping = {};
  private usedChars = new Set<string>();
  private readonly unicodeRanges = [
    [0x0370, 0x03FF],
    [0x0400, 0x04FF],
    [0x0500, 0x052F],
    [0x2C00, 0x2C5F],
    [0x0250, 0x02AF],
    [0x2200, 0x22FF]
  ];

  private readonly placeholders = new Set([
    '%current%', '%default%', '%default_entity%', '%killer_entity%',
    '%damager_entity%', '%victim_entity%', '%shooter_entity%',
    '%projectile%', '%last_entity%', '%all_mobs%', '%random_entity%',
    '%all_entities%', '%default_player%', '%killer_player%',
    '%damager_player%', '%shooter_player%', '%victim_player%',
    '%random_player%', '%all_players%', '%player%', '%selected%'
  ]);

  private readonly specialConstructPrefixes = ['%var(', '%var_local(', '%var_save(','%var_bracket(', '%math('];
  private availableChars: string[] = [];

  constructor() {
    for (const [start, end] of this.unicodeRanges) {
      for (let i = start; i <= end; i++) {
        const char = String.fromCharCode(i);
        if (char.trim()) this.availableChars.push(char);
      }
    }
  }

  private generateUniqueName(): string {
    const remaining = this.availableChars.filter(c => !this.usedChars.has(c));
    if (remaining.length > 0) {
      const char = remaining[Math.floor(Math.random() * remaining.length)];
      this.usedChars.add(char);
      return char;
    }
    let newName = '';
    let attempts = 0;
    do {
      attempts++;
      const length = 2 + Math.floor(Math.random() * 3);
      let name = '';
      for (let i = 0; i < length; i++) {
        name += this.availableChars[Math.floor(Math.random() * this.availableChars.length)];
      }
      newName = name;
    } while (this.usedChars.has(newName) && attempts < 1000);

    if (attempts >= 1000) {
      const length = 5 + Math.floor(Math.random() * 5);
      let name = 'obf_';
      for (let i = name.length; i < length; i++) {
        name += this.availableChars[Math.floor(Math.random() * this.availableChars.length)];
      }
      newName = name;
    } else {
      this.usedChars.add(newName);
    }
    return newName;
  }

  private getObfuscatedName(originalName: string): string {
    if (this.placeholders.has(originalName)) return originalName;
    if (Object.values(this.nameMapping).includes(originalName)) return originalName;

    for (const prefix of this.specialConstructPrefixes) {
      if (originalName.startsWith(prefix) && originalName.endsWith(')')) return originalName;
    }
    if (!this.nameMapping[originalName]) {
      this.nameMapping[originalName] = this.generateUniqueName();
    }
    return this.nameMapping[originalName];
  }

  private findMatchingParen(text: string, startOpenParen: number): number {
    if (text[startOpenParen] !== '(') return -1;
    let level = 1;
    for (let i = startOpenParen + 1; i < text.length; i++) {
      const char = text[i];
      if (char === '(') level++;
      else if (char === ')') {
        level--;
        if (level === 0) return i;
      }
    }
    return -1;
  }

  private obfuscateSpecialConstruct(construct: string): string {
    for (const prefix of this.specialConstructPrefixes) {
      if (construct.startsWith(prefix) && construct.endsWith(')')) {
        const openParenIndex = prefix.length - 1;
        const closeParenIndex = construct.lastIndexOf(')');
        if (openParenIndex < closeParenIndex) {
          const content = construct.substring(openParenIndex + 1, closeParenIndex);
          if (prefix === '%var_save(') return `${prefix.substring(0, prefix.length - 1)}(${content})`;
          if (prefix === '%math(') return `${prefix.substring(0, prefix.length - 1)}(${this.obfuscateMathContent(content)})`;
          const obfuscatedContent = this.getObfuscatedName(content);
          return `${prefix.substring(0, prefix.length - 1)}(${obfuscatedContent})`;
        }
      }
    }
    return construct;
  }

  private obfuscateMathContent(content: string): string {
    let result = '';
    let i = 0;
    while (i < content.length) {
      let replaced = false;
      for (const prefix of ['%var(', '%var_local(']) {
        if (content.startsWith(prefix, i)) {
          const openParenIndex = i + prefix.length - 1;
          const closeParenIndex = this.findMatchingParen(content, openParenIndex);
          if (closeParenIndex !== -1) {
            const innerContent = content.substring(openParenIndex + 1, closeParenIndex);
            const obfuscatedInner = this.getObfuscatedName(innerContent);
            result += `${prefix.substring(0, prefix.length - 1)}(${obfuscatedInner})`;
            i = closeParenIndex + 1;
            replaced = true;
            break;
          }
        }
      }
      if (replaced) continue;
      result += content[i];
      i++;
    }
    return result;
  }

  private obfuscateTextWithPlaceholders(text: string): string {
    if (typeof text !== 'string') return text as any;

    let result = '';
    let i = 0;

    while (i < text.length) {
      let replaced = false;

      for (const prefix of this.specialConstructPrefixes) {
        if (text.startsWith(prefix, i)) {
          const openParenIndex = i + prefix.length - 1;
          const closeParenIndex = this.findMatchingParen(text, openParenIndex);
          if (closeParenIndex !== -1) {
            const content = text.substring(openParenIndex + 1, closeParenIndex);
            if (prefix === '%var_save(') {
              result += `${prefix.substring(0, prefix.length - 1)}(${content})`;
            } else if (prefix === '%math(') {
              result += `${prefix.substring(0, prefix.length - 1)}(${this.obfuscateMathContent(content)})`;
            } else {
              const obf = this.getObfuscatedName(content);
              result += `${prefix.substring(0, prefix.length - 1)}(${obf})`;
            }
            i = closeParenIndex + 1;
            replaced = true;
            break;
          }
        }
      }
      if (replaced) continue;

      for (const placeholder of this.placeholders) {
        if (text.startsWith(placeholder, i)) {
          result += placeholder;
          i += placeholder.length;
          replaced = true;
          break;
        }
      }
      if (replaced) continue;

      result += text[i];
      i++;
    }
    return result;
  }

  private obfuscateVariableName(variableName: string): string {
    if (typeof variableName !== 'string') return variableName as any;

    const parts: string[] = [];
    const placeholderRegex = /(%[a-zA-Z_][a-zA-Z0-9_%()]*)/g;
    let match: RegExpExecArray | null;
    let lastEnd = 0;

    while ((match = placeholderRegex.exec(variableName)) !== null) {
      if (match.index > lastEnd) parts.push(variableName.substring(lastEnd, match.index));
      parts.push(match[0]);
      lastEnd = match.index + match[0].length;
    }
    if (lastEnd < variableName.length) parts.push(variableName.substring(lastEnd));

    const obfuscatedParts: string[] = [];
    for (const part of parts) {
      if (this.placeholders.has(part)) obfuscatedParts.push(part);
      else if (part.startsWith('%') && part.endsWith(')')) obfuscatedParts.push(this.obfuscateSpecialConstruct(part));
      else if (part) obfuscatedParts.push(this.getObfuscatedName(part));
      else obfuscatedParts.push(part);
    }
    return obfuscatedParts.join('');
  }

  public obfuscateJson(jsonData: any): any {
    if (typeof jsonData === 'object') {
      if (jsonData === null) return null;
      if (Array.isArray(jsonData)) return jsonData.map(item => this.obfuscateJson(item));

      const result: JsonValue = {};
      for (const [key, value] of Object.entries(jsonData)) {
        const data = jsonData as JsonValue;

        if (key === 'variable' && data.type === 'variable') {
          const scope = data.scope?.toString().toLowerCase();
          const originalName = value as string;
          if (scope === 'save') {
            result[key] = originalName;
          } else {
            if (this.placeholders.has(originalName)) {
              result[key] = originalName;
            } else {
              result[key] = this.obfuscateVariableName(originalName);
            }
          }
        } else if (key === 'name' && (data.type === 'function' || data.type === 'process')) {
          const originalName = value as string;
          result[key] = this.placeholders.has(originalName) ? originalName : this.getObfuscatedName(originalName);
        } else if (key === 'text' && data.type === 'text' && typeof value === 'string') {
          result[key] = this.obfuscateTextWithPlaceholders(value);
        } else if ((key === 'function_name' || key === 'process_name')) {
          const fn = value as JsonFunctionCall;
          if (fn?.type === 'text' && typeof fn.text === 'string') {
            const t = fn.text;
            const obf = this.placeholders.has(t) ? t : this.getObfuscatedName(t);
            result[key] = { ...fn, text: obf };
          } else {
            result[key] = value;
          }
        } else if (key === 'values' && Array.isArray(value)) {
          result[key] = (value as any[]).map(valItem => {
            if (valItem && typeof valItem === 'object' && valItem.name === 'function_name' && valItem.value) {
              const funcVal = valItem.value;
              if (funcVal.type === 'text' && typeof funcVal.text === 'string') {
                const t = funcVal.text;
                const obf = this.placeholders.has(t) ? t : this.getObfuscatedName(t);
                return { ...valItem, value: { ...funcVal, text: obf } };
              }
            } else if (valItem && typeof valItem === 'object' && valItem.type === 'function' && typeof valItem.function === 'string') {
              const f = valItem.function;
              if (!this.placeholders.has(f)) return { ...valItem, function: this.getObfuscatedName(f) };
            } else if (valItem && typeof valItem === 'object' && valItem.name === 'process_name' && valItem.value) {
              const procVal = valItem.value;
              if (procVal.type === 'text' && typeof procVal.text === 'string') {
                const t = procVal.text;
                const obf = this.placeholders.has(t) ? t : this.getObfuscatedName(t);
                return { ...valItem, value: { ...procVal, text: obf } };
              }
            }
            return this.obfuscateJson(valItem);
          });
        } else {
          result[key] = this.obfuscateJson(value);
        }
      }
      return result;
    }
    return jsonData;
  }

  public saveMapping(outputPath: string): void {
    const mappingFile = outputPath.replace(/\.json$/, OBFUSCATION_MAPPING_EXTENSION);
    fs.writeFileSync(mappingFile, JSON.stringify(this.nameMapping, null, 2), 'utf8');
  }

  public loadMapping(mappingPath: string): void {
    const content = fs.readFileSync(mappingPath, 'utf8');
    const loaded: NameMapping = JSON.parse(content);
    this.nameMapping = Object.entries(loaded).reduce((acc, [k, v]) => { acc[k] = v; return acc; }, {} as NameMapping);
  }
}

function getServerOptions(context: vscode.ExtensionContext): ServerOptions {
  const serverModule = context.asAbsolutePath('out\\server.py');
  const run: Executable = { command: getPythonCmd(), args: [serverModule], options: { cwd: context.extensionPath } };
  const debug: Executable = { command: getPythonCmd(), args: [serverModule], options: { cwd: context.extensionPath } };
  return { run, debug };
}

function getClientOptions(): LanguageClientOptions {
  const settings = getSettings();
  return {
    documentSelector: [{ scheme: 'file', language: 'justcode' }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher('**/*.jc'),
      configurationSection: 'jmcc-helper'
    },
    initializationOptions: {
      hideInlayHints: settings.get<boolean>('hideInlayHints', false),
      hideHover: settings.get<boolean>('hideHover', false),
      hideCompletion: settings.get<boolean>('hideCompletion', false),
      hideSignatureHelp: settings.get<boolean>('hideSignatureHelp', false),
      defaultCompileMode: settings.get<string>('defaultCompileActiveFileMode', 'UPLOAD'),
      clearTerminal: settings.get<boolean>('clearTerminalBeforeCommand', true)
    },
    outputChannel: vscode.window.createOutputChannel('JMCC Language Server'),
    outputChannelName: 'JMCC Language Server'
  };
}

async function checkAndUpdateAssets(context: vscode.ExtensionContext) {
  const log = vscode.window.createOutputChannel('JMCC Assets');
  log.appendLine('JMCC assets update started');

  function parseVersion(ver: string | null): number[] {
    if (!ver) return [];
    return ver.split('.').map(v => parseInt(v, 10) || 0);
  }
  function isVersionLess(localVer: string | null, remoteVer: string | null): boolean {
    const lv = parseVersion(localVer);
    const rv = parseVersion(remoteVer);
    const len = Math.max(lv.length, rv.length);
    for (let i = 0; i < len; i++) {
      const l = lv[i] ?? 0;
      const r = rv[i] ?? 0;
      if (l < r) return true;
      if (l > r) return false;
    }
    return false;
  }

  try {
    const { assetsDir, jmccDir } = getPaths(context);
    fs.mkdirSync(assetsDir, { recursive: true });
    fs.mkdirSync(jmccDir, { recursive: true });

    log.appendLine(`Assets dir: ${assetsDir}`);
    log.appendLine(`JMCC dir: ${jmccDir}`);

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
      log.appendLine('No workspace folder — skipping update.');
      return;
    }

    const config = loadOrInitConfig(workspaceFolder);
    const compilerPath = getCompilerPath(context, config);
    const compilerDir = path.dirname(compilerPath);
    const userPropsPath = path.join(compilerDir, PROPS_FILE_NAME);

    log.appendLine(`Compiler path: ${compilerPath}`);
    log.appendLine(`Props path: ${userPropsPath}`);

    let needsJmccUpdate = false;

    const remoteProps = await download(REMOTE_PROPS_URL);
    if (!remoteProps) {
      log.appendLine('Failed to download remote jmcc.properties — skip.');
      return;
    }

    const remoteDataVersion = extractDataVersion(remoteProps);
    log.appendLine(`Remote data_version: ${remoteDataVersion ?? 'N/A'}`);
    if (!remoteDataVersion) return;

    if (fs.existsSync(userPropsPath)) {
      const localContent = fs.readFileSync(userPropsPath, 'utf8');
      const localDataVersion = extractDataVersion(localContent);
      log.appendLine(`Local data_version: ${localDataVersion ?? 'N/A'}`);
      if (isVersionLess(localDataVersion, remoteDataVersion)) needsJmccUpdate = true;
    } else {
      needsJmccUpdate = true;
    }

    if (needsJmccUpdate) {
      log.appendLine('Updating JMCC compiler...');
      const jmccPy = await download(JMCC_PY_URL);
      if (jmccPy) {
        fs.writeFileSync(path.join(jmccDir, COMPILER_SCRIPT_NAME), jmccPy, 'utf8');
        log.appendLine('Downloaded jmcc.py');
      } else {
        log.appendLine('Failed to download jmcc.py');
      }
    }

    const completions = await download(COMPLETIONS_URL);
    if (completions) {
      fs.writeFileSync(path.join(assetsDir, COMPLETIONS_FILE_NAME), completions, 'utf8');
      log.appendLine('Downloaded completions.json');
    } else {
      log.appendLine('Failed to download completions.json');
    }

    const propsExistedBefore = fs.existsSync(userPropsPath);
    if (!propsExistedBefore) {
      log.appendLine('Props not found — init compiler.');
      if (!fs.existsSync(compilerPath)) throw new Error(`Compiler script not found: ${compilerPath}`);

      log.appendLine(`RUN: ${getPythonCmd()} "${compilerPath}" in cwd=${compilerDir}`);

      await new Promise<void>((resolve, reject) => {
        const child = spawn(getPythonCmd(), [compilerPath], { cwd: compilerDir, stdio: 'pipe', shell: false });

        let stderrData = '';
        let stdoutData = '';

        child.stderr?.on('data', d => stderrData += d.toString());
        child.stdout?.on('data', d => stdoutData += d.toString());

        child.on('close', async (code) => {
          log.appendLine(`jmcc.py exited with code ${code}`);
          if (stdoutData) log.appendLine(`stdout:\n${stdoutData}`);
          if (stderrData) log.appendLine(`stderr:\n${stderrData}`);

          const exists = fs.existsSync(userPropsPath) || await waitForFile(userPropsPath);
          if (exists) {
            let content = fs.readFileSync(userPropsPath, 'utf8');
            let lines = content.split(/\r?\n/);
            lines = upsertProp(lines, 'auto_update', 'True');
            lines = upsertProp(lines, 'check_beta_versions', 'True');
            lines = upsertProp(lines, 'data_version', remoteDataVersion);
            fs.writeFileSync(userPropsPath, lines.join('\n'), 'utf8');
            vscode.window.showInformationMessage('JMCC: Компилятор инициализирован.');
          }

          code === 0 ? resolve() : reject(new Error(`jmcc.py failed with code ${code}`));
        });

        child.on('error', err => {
          log.appendLine(`Spawn error: ${err.message}`);
          reject(err);
        });
      });
    }

    if (fs.existsSync(userPropsPath)) {
      const localContent = fs.readFileSync(userPropsPath, 'utf8');
      const localDataVersion = extractDataVersion(localContent);
      if (isVersionLess(localDataVersion, remoteDataVersion)) {
        let lines = localContent.split(/\r?\n/);
        lines = upsertProp(lines, 'data_version', remoteDataVersion);
        fs.writeFileSync(userPropsPath, lines.join('\n'), 'utf8');
        log.appendLine(`Версия данных обновлена до ${remoteDataVersion}`);
      }
    }
  } catch (err) {
    const msg = `JMCC assets update error: ${err instanceof Error ? err.message : String(err)}`;
    console.error(msg);
    vscode.window.showErrorMessage(msg);
  } finally {
    log.appendLine('JMCC assets update finished');
    log.show(true);
  }
}

export async function activate(context: vscode.ExtensionContext) {
  console.log('Activating JMCC Extension...');
  const { assetsDir, jmccDir } = getPaths(context);
  fs.mkdirSync(assetsDir, { recursive: true });
  fs.mkdirSync(jmccDir, { recursive: true });

  try {
    await checkAndUpdateAssets(context);
    console.log('Assets checked/updated.');

    const serverOptions = getServerOptions(context);
    const clientOptions = getClientOptions();

    client = new LanguageClient(
      'jmccLanguageServer',
      'JMCC Language Server',
      serverOptions,
      clientOptions
    );

    await client.start();
    console.log('JMCC Language Server started');
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    vscode.window.showErrorMessage(`JMCC: Ошибка запуска LSP-сервера или проверки обновлений: ${errorMessage}`);
    console.error('Activation error:', err);
  }

  const existingTerminal = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
  if (existingTerminal) existingTerminal.dispose();

  const commands = [
    vscode.commands.registerCommand('jmcc.compileAsUrl', async (uri: vscode.Uri) => {
      const folder = getWorkspaceFolder(uri);
      if (!folder) return;
      const isDir = fs.statSync(uri.fsPath).isDirectory();
      if (!isDir) await ensureDocumentSaved(uri);
      compile(uri.fsPath, 'UPLOAD', folder, context);
    }),

    vscode.commands.registerCommand('jmcc.compileAsFile', async (uri: vscode.Uri) => {
      const folder = getWorkspaceFolder(uri);
      if (!folder) return;
      const isDir = fs.statSync(uri.fsPath).isDirectory();
      if (!isDir) await ensureDocumentSaved(uri);
      compile(uri.fsPath, 'SAVE', folder, context, true);
    }),

    vscode.commands.registerCommand('jmcc.decompileFile', async (uri: vscode.Uri) => {
      const folder = getWorkspaceFolder(uri);
      if (folder) decompile(uri.fsPath, folder, context);
    }),

    vscode.commands.registerCommand('jmcc.compileActiveFile', async () => {
      const activeEditor = vscode.window.activeTextEditor;
      if (!activeEditor) return;
      const document = activeEditor.document;
      if (document.isUntitled) {
        vscode.window.showWarningMessage('Сперва сохраните файл.');
        return;
      }
      await ensureDocumentSaved(document.uri);
      const folder = getWorkspaceFolder(document.uri);
      if (folder && document.fileName.endsWith('.jc')) {
        compile(document.fileName, 'SAVE', folder, context);
      }
    }),

    vscode.commands.registerCommand('jmcc.runActiveFile', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const doc = editor.document;
      if (doc.isUntitled) {
        vscode.window.showWarningMessage('Сперва сохраните файл.');
        return;
      }
      await ensureDocumentSaved(doc.uri);
      const filePath = doc.fileName;
      const folder = getWorkspaceFolder(doc.uri);
      if (!folder) return;

      const mode = getDefaultCompileMode();

      if (filePath.endsWith('.jc')) {
        if (mode === 'OBFUSCATE SAVE') {
          await compileWithObfuscation(filePath, 'FILE', folder, context);
        } else if (mode === 'OBFUSCATE URL') {
          await compileWithObfuscation(filePath, 'URL', folder, context);
        } else {
          compile(filePath, mode, folder, context);
        }
      } else if (filePath.endsWith('.json')) {
        decompile(filePath, folder, context);
      }
    }),

    vscode.commands.registerCommand('jmcc.compileObfuscateFile', (uri: vscode.Uri) => {
      const folder = getWorkspaceFolder(uri);
      if (folder) compileWithObfuscation(uri.fsPath, 'FILE', folder, context);
    }),

    vscode.commands.registerCommand('jmcc.compileObfuscateUrl', (uri: vscode.Uri) => {
      const folder = getWorkspaceFolder(uri);
      if (folder) compileWithObfuscation(uri.fsPath, 'URL', folder, context);
    }),

    vscode.commands.registerCommand('jmcc.decompileFileWithObfmap', (uri: vscode.Uri) => {
      const folder = getWorkspaceFolder(uri);
      if (folder) decompileWithObfmap(uri.fsPath, folder, context);
    }),

    vscode.commands.registerCommand('jmcc.saveAndUpload', async (uri: vscode.Uri) => {
      const folder = getWorkspaceFolder(uri);
      if (folder) {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor && activeEditor.document.uri.fsPath === uri.fsPath && activeEditor.document.isDirty) {
          await activeEditor.document.save();
        }
        saveAndUpload(uri.fsPath, folder, context);
      }
    })
  ];

  context.subscriptions.push(...commands);
  console.log('JMCC Extension activated.');
}

export async function deactivate(): Promise<void> {
  console.log('Deactivating JMCC Extension...');
  if (client) {
    await client.stop();
    client = undefined;
  }
  console.log('JMCC Extension deactivated.');
}
