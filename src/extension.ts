import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { LanguageClient, LanguageClientOptions, ServerOptions, Executable } from 'vscode-languageclient/node';
import * as https from 'https';
import { spawn, ChildProcess } from 'child_process';

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


async function ensureDocumentSaved(uri: vscode.Uri): Promise<void> {
  const doc = await vscode.workspace.openTextDocument(uri);
  if (doc.isDirty) {
    await doc.save();
  }
}

function shouldClearTerminal(): boolean {
  return getSettings().get<boolean>('clearTerminalBeforeCommand', true);
}

function getDefaultCompileMode(): 'UPLOAD' | 'SAVE' | 'BOTH' {
  return getSettings()
    .get<string>('defaultCompileActiveFileMode', 'UPLOAD')
    .toUpperCase() as 'UPLOAD' | 'SAVE' | 'BOTH';
}

function getSettings(): vscode.WorkspaceConfiguration {
  
  return vscode.workspace.getConfiguration('jmcc-helper');
}

function getConfigPath(workspaceFolder: vscode.WorkspaceFolder): string {
    const configDir = path.join(workspaceFolder.uri.fsPath, '.vscode');
    if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
    }
    return path.join(configDir, CONFIG_FILE_NAME);
}

function loadOrInitConfig(workspaceFolder: vscode.WorkspaceFolder): JMCCConfig {
  const configPath = getConfigPath(workspaceFolder);
  const defaultConfig: JMCCConfig = {
    compilerPath: '',
    compilerOutputPath: ''
  };
  const autoCreateConfig = getSettings().get<boolean>('autoCreateConfig', true);
  if (!fs.existsSync(configPath)) {
    if (autoCreateConfig) {
      fs.writeFileSync(configPath, JSON.stringify(defaultConfig, null, 2), 'utf-8');
    }
    return defaultConfig;
  }
  let config: JMCCConfig;
  try {
    config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  } catch (err) {
    vscode.window.showErrorMessage(
      `JMCC: Ошибка чтения ${path.basename(configPath)}. Проверьте формат JSON.`
    );
    throw err;
  }
  let modified = false;
  if (typeof config.compilerPath !== 'string') {
    config.compilerPath = '';
    modified = true;
  }
  if (typeof config.compilerOutputPath !== 'string') {
    config.compilerOutputPath = '';
    modified = true;
  }
  if (modified && autoCreateConfig) {
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');
  }

  return config;
}


function getCompilerPath(context: vscode.ExtensionContext, config: JMCCConfig): string {
  const settingPath = getSettings().get<string>('compilerPath', '').trim();

  if (config.compilerPath?.trim()) {
    return path.normalize(config.compilerPath);
  }
  if (settingPath) {
    return path.normalize(settingPath);
  }
  return context.asAbsolutePath(path.join('out', JMCC_DIR_NAME, COMPILER_SCRIPT_NAME));
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

function validateCompilerPath(compilerPath: string, context: vscode.ExtensionContext, config: JMCCConfig): boolean {
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

function runCommand(command: string, workspaceFolder: vscode.WorkspaceFolder) {
  const terminal = getOrCreateTerminal();
  terminal.show();

  if (shouldClearTerminal()) {
    const clearCmd = os.platform() === 'win32' ? 'cls' : 'clear';
    terminal.sendText(clearCmd);
  }

  terminal.sendText(command);
}

function buildBaseCommandArgs(compilerPath: string, targetPath: string): string[] {
    const pythonCommand = os.platform() === 'win32' ? 'py' : 'python3';
    return [pythonCommand, `"${compilerPath}"`, 'compile', `"${path.normalize(targetPath)}"`];
}

function getWorkspaceFolder(fileUri: vscode.Uri): vscode.WorkspaceFolder | undefined {
    return vscode.workspace.getWorkspaceFolder(fileUri);
}

function compile(targetPath: string, mode: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext, isCompileAsFile: boolean = false) {
    const config = loadOrInitConfig(workspaceFolder);
    const compilerPath = getCompilerPath(context, config);

    if (!validateCompilerPath(compilerPath, context, config)) {
        return;
    }

    const args = buildBaseCommandArgs(compilerPath, targetPath);

    if (mode === 'UPLOAD') {
        args.push('-u');
    } else if (mode === 'BOTH') {
        args.push('-su');
    }

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

    if (!validateCompilerPath(compilerPath, context, config)) {
        return;
    }

    const pythonCommand = os.platform() === 'win32' ? 'py' : 'python3';
    const args = [pythonCommand, `"${compilerPath}"`, 'decompile', `"${path.normalize(targetPath)}"`];
    runCommand(args.join(' '), workspaceFolder);
}

function saveAndUpload(targetPath: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext) {
    const config = loadOrInitConfig(workspaceFolder);
    const compilerPath = getCompilerPath(context, config);

    if (!validateCompilerPath(compilerPath, context, config)) {
        return;
    }

    const pythonCommand = os.platform() === 'win32' ? 'py' : 'python3';
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

        console.log('Stage 1: Compiling to JSON...');
        const compileCommand = `py "${compilerPath}" compile "${filePath}"`;
        await executeCompilation(compileCommand, workspaceFolder.uri.fsPath);

        let attempts = 0;
        while (!fs.existsSync(outputPath) && attempts < 20) {
            await new Promise(resolve => setTimeout(resolve, 100));
            attempts++;
        }
        if (!fs.existsSync(outputPath)) {
            throw new Error(`Stage 1: JSON file not found after compilation: ${outputPath}`);
        }

        console.log('Stage 2: Obfuscating...');
        const jsonContent = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
        const obfuscator = new Obfuscator();
        const obfuscatedJson = obfuscator.obfuscateJson(jsonContent);
        fs.writeFileSync(outputPath, JSON.stringify(obfuscatedJson, null, 2));
        obfuscator.saveMapping(outputPath);

        if (mode === 'URL') {
            console.log('Stage 3: Decompiling obfuscated JSON...');
            const decompileCommand = `py "${compilerPath}" decompile "${outputPath}"`;
            await executeCompilation(decompileCommand, workspaceFolder.uri.fsPath);

            attempts = 0;
            while (!fs.existsSync(decompiledPath) && attempts < 20) {
                await new Promise(resolve => setTimeout(resolve, 100));
                attempts++;
            }
            if (!fs.existsSync(decompiledPath)) {
                throw new Error(`Stage 3: Decompiled file not found: ${decompiledPath}`);
            }

            console.log('Stage 4: Preparing compile + cleanup command for terminal...');
            const terminal = getOrCreateTerminal();
            const shellName = terminal.name.toLowerCase();
            const isPowerShell = /powershell|pwsh/.test(terminal.processId?.toString() || shellName);
            const finalCompileCmd = `py "${compilerPath}" compile "${decompiledPath}" -u`;

            let cleanupCmd: string;
            if (isPowerShell || process.platform === 'win32') {
                const escapedJson = outputPath.replace(/"/g, '`"');
                const escapedJc = decompiledPath.replace(/"/g, '`"');
                cleanupCmd = `${finalCompileCmd}; if ($?) { del "${escapedJson}"; del "${escapedJc}" }`;
            } else {
                const escapedJson = outputPath.includes(' ') ? `"${outputPath}"` : outputPath;
                const escapedJc = decompiledPath.includes(' ') ? `"${decompiledPath}"` : decompiledPath;
                cleanupCmd = `${finalCompileCmd} && rm ${escapedJson} && rm ${escapedJc}`;
            }

            terminal.show();
            terminal.sendText(cleanupCmd, true);
        }

        vscode.window.showInformationMessage(`Obfuscation and compilation completed successfully for ${filePath}`);
    } catch (error: any) {
        const errorMessage = error.message || 'Unknown error';
        vscode.window.showErrorMessage(`Compilation/Obfuscation failed: ${errorMessage}`);
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
        const errorMessage = error.message || error.toString();
        vscode.window.showErrorMessage(`Error during deobfuscation: ${errorMessage}`);
    }
}

async function executeCompilation(command: string, cwd: string): Promise<string> {
    return new Promise((resolve, reject) => {
        const child: ChildProcess = spawn(command, [], { shell: true, cwd });
        let stdout = '';
        let stderr = '';

        child.stdout?.on('data', (data) => stdout += data);
        child.stderr?.on('data', (data) => stderr += data);

        child.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Compilation failed with code ${code}: ${stderr}`));
            } else {
                resolve(stdout);
            }
        });

        child.on('error', (err) => {
            reject(new Error(`Spawn error: ${err.message}`));
        });
    });
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
        [0x2200, 0x22FF],
    ];
    
    private readonly placeholders = new Set([
        '%current%', '%default%', '%default_entity%', '%killer_entity%',
        '%damager_entity%', '%victim_entity%', '%shooter_entity%',
        '%projectile%', '%last_entity%', '%all_mobs%', '%random_entity%',
        '%all_entities%', '%default_player%', '%killer_player%',
        '%damager_player%', '%shooter_player%', '%victim_player%',
        '%random_player%', '%all_players%', '%player%', '%selected%'
    ]);
    
    private readonly specialConstructPrefixes = ['%var(', '%var_local(', '%var_save(', '%math('];
    private availableChars: string[] = [];
    
    constructor() {
        for (const [start, end] of this.unicodeRanges) {
            for (let i = start; i <= end; i++) {
                const char = String.fromCharCode(i);
                if (char.trim()) {
                    this.availableChars.push(char);
                }
            }
        }
    }
    
    private generateUniqueName(): string {
        const remainingChars = this.availableChars.filter(c => !this.usedChars.has(c));
        if (remainingChars.length > 0) {
            const char = remainingChars[Math.floor(Math.random() * remainingChars.length)];
            this.usedChars.add(char);
            return char;
        } else {
            let newName: string;
            let attempts = 0;
            const maxAttempts = 1000;
            do {
                attempts++;
                const length = 2 + Math.floor(Math.random() * 3);
                let name = '';
                for (let i = 0; i < length; i++) {
                    name += this.availableChars[Math.floor(Math.random() * this.availableChars.length)];
                }
                newName = name;
            } while (this.usedChars.has(newName) && attempts < maxAttempts);
            if (attempts >= maxAttempts) {
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
    }
    
    private getObfuscatedName(originalName: string): string {
        if (this.placeholders.has(originalName)) {
            return originalName;
        }
        if (Object.values(this.nameMapping).includes(originalName)) {
            return originalName;
        }
        for (const prefix of this.specialConstructPrefixes) {
            if (originalName.startsWith(prefix) && originalName.endsWith(')')) {
                return originalName;
            }
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
            if (char === '(') {
                level++;
            } else if (char === ')') {
                level--;
                if (level === 0) {
                    return i;
                }
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
                    if (prefix === '%var_save(') {
                        return prefix.substring(0, prefix.length - 1) + '(' + content + ')';
                    }
                    if (prefix === '%math(') {
                        return prefix.substring(0, prefix.length - 1) + '(' + this.obfuscateMathContent(content) + ')';
                    }
                    const obfuscatedContent = this.getObfuscatedName(content);
                    return prefix.substring(0, prefix.length - 1) + '(' + obfuscatedContent + ')';
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
                        const contentStart = openParenIndex + 1;
                        const innerContent = content.substring(contentStart, closeParenIndex);
                        const obfuscatedInner = this.getObfuscatedName(innerContent);
                        const fullConstruct = prefix.substring(0, prefix.length - 1) + '(' + obfuscatedInner + ')';
                        result += fullConstruct;
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
        if (typeof text !== 'string') return text;
        
        let result = '';
        let i = 0;
        
        while (i < text.length) {
            let replaced = false;
            
            for (const prefix of this.specialConstructPrefixes) {
                if (text.startsWith(prefix, i)) {
                    const openParenIndex = i + prefix.length - 1;
                    const closeParenIndex = this.findMatchingParen(text, openParenIndex);
                    if (closeParenIndex !== -1) {
                        const contentStart = openParenIndex + 1;
                        const content = text.substring(contentStart, closeParenIndex);
                        if (prefix === '%var_save(') {
                            const fullConstruct = prefix.substring(0, prefix.length - 1) + '(' + content + ')';
                            result += fullConstruct;
                            i = closeParenIndex + 1;
                            replaced = true;
                            break;
                        }
                        if (prefix === '%math(') {
                            const fullConstruct = prefix.substring(0, prefix.length - 1) + '(' + this.obfuscateMathContent(content) + ')';
                            result += fullConstruct;
                            i = closeParenIndex + 1;
                            replaced = true;
                            break;
                        }
                        const obfuscatedContent = this.getObfuscatedName(content);
                        const fullConstruct = prefix.substring(0, prefix.length - 1) + '(' + obfuscatedContent + ')';
                        result += fullConstruct;
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
        if (typeof variableName !== 'string') return variableName;
        
        const parts: string[] = [];
        let currentIndex = 0;
        
        const placeholderRegex = /(%[a-zA-Z_][a-zA-Z0-9_%()]*)/g;
        let match;
        let lastEnd = 0;
        
        while ((match = placeholderRegex.exec(variableName)) !== null) {
            if (match.index > lastEnd) {
                parts.push(variableName.substring(lastEnd, match.index));
            }
            parts.push(match[0]);
            lastEnd = match.index + match[0].length;
        }
        
        if (lastEnd < variableName.length) {
            parts.push(variableName.substring(lastEnd));
        }
        
        const obfuscatedParts: string[] = [];
        for (const part of parts) {
            if (this.placeholders.has(part)) {
                obfuscatedParts.push(part);
            } else if (part.startsWith('%') && part.endsWith(')')) {
                obfuscatedParts.push(this.obfuscateSpecialConstruct(part));
            } else if (part) {
                obfuscatedParts.push(this.getObfuscatedName(part));
            } else {
                obfuscatedParts.push(part);
            }
        }
        
        return obfuscatedParts.join('');
    }
    
    public obfuscateJson(jsonData: any): any {
        if (typeof jsonData === 'object') {
            if (jsonData === null) return null;
            if (Array.isArray(jsonData)) {
                return jsonData.map(item => this.obfuscateJson(item));
            }
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
                    if (this.placeholders.has(originalName)) {
                        result[key] = originalName;
                    } else {
                        result[key] = this.getObfuscatedName(originalName);
                    }
                } else if (key === 'text' && data.type === 'text' && typeof value === 'string') {
                    result[key] = this.obfuscateTextWithPlaceholders(value);
                } else if ((key === 'function_name' || key === 'process_name')) {
                    const functionCall = value as JsonFunctionCall;
                    if (functionCall?.type === 'text' && typeof functionCall.text === 'string') {
                        const originalText = functionCall.text;
                        let obfuscatedText = originalText;
                        if (this.placeholders.has(originalText)) {
                            obfuscatedText = originalText;
                        } else {
                            obfuscatedText = this.getObfuscatedName(originalText);
                        }
                        result[key] = {
                            ...functionCall,
                            text: obfuscatedText
                        };
                    } else {
                        result[key] = value;
                    }
                } else if (key === 'values' && Array.isArray(value)) {
                    result[key] = (value as any[]).map(valItem => {
                        if (valItem && typeof valItem === 'object' && valItem.name === 'function_name' && valItem.value) {
                            const funcVal = valItem.value;
                            if (funcVal.type === 'text' && typeof funcVal.text === 'string') {
                                const originalText = funcVal.text;
                                let obfuscatedText = originalText;
                                if (this.placeholders.has(originalText)) {
                                    obfuscatedText = originalText;
                                } else {
                                    obfuscatedText = this.getObfuscatedName(originalText);
                                }
                                return {
                                    ...valItem,
                                    value: {
                                        ...funcVal,
                                        text: obfuscatedText
                                    }
                                };
                            }
                        } else if (valItem && typeof valItem === 'object' && valItem.type === 'function' && typeof valItem.function === 'string') {
                            const originalFuncName = valItem.function;
                            if (this.placeholders.has(originalFuncName)) {
                                return valItem;
                            } else {
                                return {
                                    ...valItem,
                                    function: this.getObfuscatedName(originalFuncName)
                                };
                            }
                        } else if (valItem && typeof valItem === 'object' && valItem.name === 'process_name' && valItem.value) {
                            const procVal = valItem.value;
                            if (procVal.type === 'text' && typeof procVal.text === 'string') {
                                const originalText = procVal.text;
                                let obfuscatedText = originalText;
                                if (this.placeholders.has(originalText)) {
                                    obfuscatedText = originalText;
                                } else {
                                    obfuscatedText = this.getObfuscatedName(originalText);
                                }
                                return {
                                    ...valItem,
                                    value: {
                                        ...procVal,
                                        text: obfuscatedText
                                    }
                                };
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
        try {
            const content = fs.readFileSync(mappingPath, 'utf8');
            const loadedMapping: NameMapping = JSON.parse(content);
            this.nameMapping = Object.entries(loadedMapping).reduce((acc, [key, value]) => {
                acc[key] = value;
                return acc;
            }, {} as NameMapping);
        } catch (error: any) {
            const errorMessage = error.message || error.toString();
            throw new Error(`Failed to load mapping file: ${errorMessage}`);
        }
    }
}

function getServerOptions(context: vscode.ExtensionContext): ServerOptions {
    const pythonCommand = os.platform() === 'win32' ? 'py' : 'python3';
    const serverModule = context.asAbsolutePath(path.join('out', 'server.py'));
    const run: Executable = {
        command: pythonCommand,
        args: [serverModule],
        options: { cwd: context.extensionPath }
    };
    const debug: Executable = {
        command: pythonCommand,
        args: [serverModule],
        options: { cwd: context.extensionPath }
    };
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
      hideInlayHints:    settings.get<boolean>('hideInlayHints', false),
      hideHover:         settings.get<boolean>('hideHover', false),
      hideCompletion:    settings.get<boolean>('hideCompletion', false),
      hideSignatureHelp: settings.get<boolean>('hideSignatureHelp', false),
      defaultCompileMode: settings.get<string>('defaultCompileActiveFileMode', 'UPLOAD'),
      clearTerminal:     settings.get<boolean>('clearTerminalBeforeCommand', true)
    },
    outputChannel: vscode.window.createOutputChannel('JMCC Language Server'),
    outputChannelName: 'JMCC Language Server'
  };
}

async function download(url: string): Promise<string | null> {
    return new Promise((resolve, reject) => {
        https.get(url.trim(), res => {
            if (res.statusCode !== 200) {
                console.warn(`Download failed for ${url}: Status ${res.statusCode}`);
                return resolve(null);
            }
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve(data));
        }).on('error', err => {
            console.error('Download error:', err);
            resolve(null);
        });
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
    if (!found) {
        updated.push(`${key} = ${value}`);
    }
    return updated;
}

async function waitForFile(filePath: string, timeoutMs: number = 3000): Promise<boolean> {
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

async function checkAndUpdateAssets(context: vscode.ExtensionContext) {
    const assetsDir = context.asAbsolutePath(path.join('out', ASSETS_DIR_NAME));
    const jmccDir = context.asAbsolutePath(path.join('out', JMCC_DIR_NAME));

    if (!fs.existsSync(assetsDir)) fs.mkdirSync(assetsDir, { recursive: true });
    if (!fs.existsSync(jmccDir)) fs.mkdirSync(jmccDir, { recursive: true });

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        console.warn("No workspace folder found, skipping asset update.");
        return;
    }

    const config = loadOrInitConfig(workspaceFolder);
    const compilerPath = getCompilerPath(context, config);
    const compilerDir = path.dirname(compilerPath);
    const userPropsPath = path.join(compilerDir, PROPS_FILE_NAME);

    let needsJmccUpdate = false;

    const REMOTE_PROPS_CONTENT = await download(REMOTE_PROPS_URL);
    if (!REMOTE_PROPS_CONTENT) {
        console.warn("Could not download jmcc.properties, skipping update check.");
        return;
    }

    const remoteDataVersion = extractDataVersion(REMOTE_PROPS_CONTENT);
    if (!remoteDataVersion) {
        console.warn("Could not extract data_version from remote jmcc.properties.");
        return;
    }

    if (fs.existsSync(userPropsPath)) {
        const localContent = fs.readFileSync(userPropsPath, 'utf8');
        const localDataVersion = extractDataVersion(localContent);
        if (localDataVersion !== remoteDataVersion) {
            needsJmccUpdate = true;
        }
    } else {
        needsJmccUpdate = true;
    }

    if (needsJmccUpdate) {
        console.log("Updating JMCC compiler...");
        const jmccPyContent = await download(JMCC_PY_URL);
        if (jmccPyContent) {
            fs.writeFileSync(path.join(jmccDir, COMPILER_SCRIPT_NAME), jmccPyContent, 'utf8');
            console.log('Downloaded: jmcc.py');
        } else {
            console.error("Failed to download jmcc.py");
        }
    }

    const completionsContent = await download(COMPLETIONS_URL);
    if (completionsContent) {
        fs.writeFileSync(path.join(assetsDir, COMPLETIONS_FILE_NAME), completionsContent, 'utf8');
        console.log('Downloaded: completions.json');
    } else {
        console.error("Failed to download completions.json");
    }

const propsExistedBefore = fs.existsSync(userPropsPath);
if (!propsExistedBefore) {
    vscode.window.showInformationMessage('JMCC: Инициализация компилятора... Запуск jmcc.py');
    const pythonCommand = os.platform() === 'win32' ? 'py' : 'python3';

    try {
        await new Promise<void>((resolve, reject) => {
            const child = spawn(pythonCommand, [compilerPath], {
                cwd: compilerDir,
                stdio: 'pipe',
                shell: false
            });

            let stderrData = '';
            let stdoutData = '';

            child.stderr?.on('data', data => stderrData += data.toString());
            child.stdout?.on('data', data => stdoutData += data.toString());

            child.on('close', async (code) => {
                const exists = fs.existsSync(userPropsPath) || await waitForFile(userPropsPath);
                
                if (exists) {
                    let content = fs.readFileSync(userPropsPath, 'utf8');
                    let lines = content.split(/\r?\n/);
                    lines = upsertProp(lines, 'auto_update', 'True');
                    lines = upsertProp(lines, 'check_beta_versions', 'True');
                    lines = upsertProp(lines, 'data_version', remoteDataVersion);
                    fs.writeFileSync(userPropsPath, lines.join('\n'), 'utf8');
                    vscode.window.showInformationMessage('JMCC: Компилятор успешно инициализирован и настроен.');
                } else {
                    return;
                }

                if (code === 0) {
                    console.log('jmcc.py executed successfully.');
                    console.log('stdout:', stdoutData);
                    resolve();
                } else {
                    console.error(`jmcc.py failed with code ${code}`);
                    console.error('stderr:', stderrData);
                    reject(new Error(`jmcc.py failed with code ${code}`));
                }
            });

            child.on('error', err => {
                console.error('Spawn error:', err);
                reject(err);
            });
        });
    } catch (err) {
        console.error('Error during jmcc.py execution:', err);
        vscode.window.showErrorMessage('JMCC: Ошибка при запуске компилятора: ' + (err as Error).message);
    }
}

    if (fs.existsSync(userPropsPath)) {
        const localContent = fs.readFileSync(userPropsPath, 'utf8');
        const localDataVersion = extractDataVersion(localContent);
        if (localDataVersion !== remoteDataVersion) {
            let lines = localContent.split(/\r?\n/);
            lines = upsertProp(lines, 'data_version', remoteDataVersion);
            fs.writeFileSync(userPropsPath, lines.join('\n'), 'utf8');
            console.log(`Updated data_version in ${userPropsPath} to ${remoteDataVersion}`);
        }
    }
}

export async function activate(context: vscode.ExtensionContext) {
    console.log('Activating JMCC Extension...');

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
        console.error("Activation error:", err);
    }

    const existingTerminal = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
    if (existingTerminal) {
        existingTerminal.dispose();
    }
    const commands = [
        vscode.commands.registerCommand('jmcc.compileAsFile', async (uri: vscode.Uri) => {
            await ensureDocumentSaved(uri);
            const folder = getWorkspaceFolder(uri);
            if (folder) compile(uri.fsPath, 'SAVE', folder, context, true);
        }),
        vscode.commands.registerCommand('jmcc.compileAsUrl', async (uri: vscode.Uri) => {
            await ensureDocumentSaved(uri);
            const folder = getWorkspaceFolder(uri);
            if (folder) compile(uri.fsPath, 'UPLOAD', folder, context);
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
                 vscode.window.showWarningMessage("Please save the file first.");
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
    vscode.window.showWarningMessage('Please save the file first.');
    return;
  }
  await ensureDocumentSaved(doc.uri);
  const filePath = doc.fileName;
  const folder = getWorkspaceFolder(doc.uri);
  if (!folder) return;

  if (filePath.endsWith('.jc')) {
    const mode = getDefaultCompileMode();
    compile(filePath, mode, folder, context);
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
