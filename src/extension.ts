import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { LanguageClient, LanguageClientOptions, ServerOptions, Executable } from 'vscode-languageclient/node';
import * as https from 'https';

const TERMINAL_NAME = 'JMCC Terminal';
let client: LanguageClient | undefined;

async function checkAndUpdateAssets(context: vscode.ExtensionContext) {
    const assetsDir = context.asAbsolutePath('out/assets');
    const jmccDir = context.asAbsolutePath('out/JMCC');
    if (!fs.existsSync(assetsDir)) {
        fs.mkdirSync(assetsDir, { recursive: true });
    }
    if (!fs.existsSync(jmccDir)) {
        fs.mkdirSync(jmccDir, { recursive: true });
    }

    const localPropsPath = path.join(jmccDir, 'jmcc.properties');
    const REMOTE_PROPS_URL = 'https://raw.githubusercontent.com/donzgold/JustMC_compilator/master/jmcc.properties';
    const COMPLETIONS_URL = 'https://raw.githubusercontent.com/donzgold/JustMC_compilator/master/assets/completions.json';
    const HOVER_URL = 'https://raw.githubusercontent.com/donzgold/JustMC_compilator/master/assets/hover.json';
    const JMCC_PY_URL = 'https://raw.githubusercontent.com/donzgold/JustMC_compilator/master/jmcc.py';

    async function download(url: string): Promise<string | null> {
        return new Promise(resolve => {
            https.get(url, res => {
                if (res.statusCode !== 200) return resolve(null);
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => resolve(data));
            }).on('error', () => resolve(null));
        });
    }

    function extractDataVersion(content: string): string | null {
        const match = content.match(/^\s*data_version\s*=\s*(\S+)/m);
        return match ? match[1].trim() : null;
    }

    const remotePropsContent = await download(REMOTE_PROPS_URL);
    if (!remotePropsContent) return;
    const remoteDataVersion = extractDataVersion(remotePropsContent);
    if (!remoteDataVersion) return;

    const hasLocalProps = fs.existsSync(localPropsPath);
    let localDataVersion: string | null = null;

    if (hasLocalProps) {
        const content = fs.readFileSync(localPropsPath, 'utf8');
        localDataVersion = extractDataVersion(content);
    }

    let updated = false;

    if (!hasLocalProps) {
        const files = [
            { url: COMPLETIONS_URL, path: path.join(assetsDir, 'completions.json') },
            { url: HOVER_URL, path: path.join(assetsDir, 'hover.json') },
            { url: JMCC_PY_URL, path: path.join(jmccDir, 'jmcc.py') }
        ];

        for (const file of files) {
            const content = await download(file.url);
            if (content) {
                fs.writeFileSync(file.path, content, 'utf8');
            }
        }

        fs.writeFileSync(localPropsPath, `data_version = ${remoteDataVersion}\n`, 'utf8');
        updated = true;
    } else if (localDataVersion !== remoteDataVersion) {
        const lines = fs.readFileSync(localPropsPath, 'utf8').split(/\r?\n/);
        const updatedLines = lines.map(line =>
            /^\s*data_version\s*=/.test(line)
                ? `data_version = ${remoteDataVersion}`
                : line
        );
        fs.writeFileSync(localPropsPath, updatedLines.join('\n'), 'utf8');

        const assets = [
            { url: COMPLETIONS_URL, path: path.join(assetsDir, 'completions.json') },
            { url: HOVER_URL, path: path.join(assetsDir, 'hover.json') },
            { url: JMCC_PY_URL, path: path.join(jmccDir, 'jmcc.py') }
        ];

        for (const asset of assets) {
            const content = await download(asset.url);
            if (content) {
                fs.writeFileSync(asset.path, content, 'utf8');
            }
        }

        updated = true;
    }

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (workspaceFolder) {
        const config = loadOrInitConfig(workspaceFolder);
        const userCompilerPath = config?.compilerPath?.trim();

        if (userCompilerPath) {
            const userDir = path.dirname(userCompilerPath);
            const userPyPath = path.join(userDir, 'jmcc.py');
            const userPropsPath = path.join(userDir, 'jmcc.properties');

            const jmccPyContent = await download(JMCC_PY_URL);
            if (jmccPyContent && fs.existsSync(userPyPath)) {
                fs.writeFileSync(userPyPath, jmccPyContent, 'utf8');
            }

            if (fs.existsSync(userPropsPath)) {
                const lines = fs.readFileSync(userPropsPath, 'utf8').split(/\r?\n/);
                const updatedUser = lines.map(line =>
                    /^\s*data_version\s*=/.test(line)
                        ? `data_version = ${remoteDataVersion}`
                        : line
                ).join('\n');
                fs.writeFileSync(userPropsPath, updatedUser, 'utf8');
            }
        }
    }

    if (updated) {
        const choice = await vscode.window.showInformationMessage(
            `JMCC: Обновлены данные до версии ${remoteDataVersion}. Перезапустить сервер?`,
            'Да', 'Нет'
        );
        if (choice === 'Да' && client) {
            await client.stop();
            await client.start();
            vscode.window.showInformationMessage('JMCC: Сервер перезапущен с новыми данными.');
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
    const outputChannel = vscode.window.createOutputChannel('JMCC Language Server');
    return {
        documentSelector: [{ scheme: 'file', language: 'justcode' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.jc')
        },
        outputChannel: outputChannel,
        outputChannelName: 'JMCC Language Server'
    };
}

function getOrCreateTerminal(): vscode.Terminal {
    const existing = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
    return existing ?? vscode.window.createTerminal(TERMINAL_NAME);
}

function getConfigPath(workspaceFolder: vscode.WorkspaceFolder): string {
    const configDir = path.join(workspaceFolder.uri.fsPath, '.vscode');
    if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
    }
    return path.join(configDir, '.jmccconfig.json');
}

function loadOrInitConfig(workspaceFolder: vscode.WorkspaceFolder): any {
    const configPath = getConfigPath(workspaceFolder);
    if (!fs.existsSync(configPath)) {
        const defaultConfig = {
            compilerPath: "",
            defaultCompileActiveFileMode: "UPLOAD",
            compilerOutputPath: "",
            clearTerminalBeforeCommand: true
        };
        fs.writeFileSync(configPath, JSON.stringify(defaultConfig, null, 2), 'utf-8');
        vscode.window.showWarningMessage(`Укажите путь к компилятору Python в файле: ${configPath}`);
    }
    let config;
    try {
        config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    } catch (err) {
        vscode.window.showErrorMessage('JMCC: Ошибка чтения .jmccconfig.json. Проверьте формат JSON.');
        throw err;
    }
    let modified = false;
    if (!config.defaultCompileActiveFileMode) {
        config.defaultCompileActiveFileMode = 'UPLOAD';
        modified = true;
    } else if (config.defaultCompileActiveFileMode.toUpperCase() === 'FILE') {
        config.defaultCompileActiveFileMode = 'SAVE';
        modified = true;
    }
    if (!config.compilerOutputPath) {
        config.compilerOutputPath = '';
        modified = true;
    }
    if (config.clearTerminalBeforeCommand === undefined) {
        config.clearTerminalBeforeCommand = true;
        modified = true;
    }
    if (modified) {
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2), 'utf-8');
    }
    return config;
}

function getWorkspaceFolder(fileUri: vscode.Uri): vscode.WorkspaceFolder | undefined {
    return vscode.workspace.getWorkspaceFolder(fileUri);
}

function runCommand(command: string, workspaceFolder: vscode.WorkspaceFolder) {
    const config = loadOrInitConfig(workspaceFolder);
    const terminal = getOrCreateTerminal();
    terminal.show();
    if (config.clearTerminalBeforeCommand !== false) {
        terminal.sendText(os.platform() === 'win32' ? 'cls' : 'clear');
    }
    terminal.sendText(command);
}

function validatePath(filePath: string, isFile: boolean = false): boolean {
    try {
        if (!fs.existsSync(filePath)) return false;
        const stat = fs.statSync(filePath);
        if (isFile && stat.isDirectory()) return false;
        return true;
    } catch (err) {
        return false;
    }
}

function getCompilerPath(context: vscode.ExtensionContext, config: any): string {
    if (config.compilerPath && config.compilerPath.trim() !== '') {
        return path.normalize(config.compilerPath);
    }
    return context.asAbsolutePath(path.join('out', 'JMCC', 'jmcc.py'));
}

function compile(targetPath: string, mode: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext, isCompileAsFile: boolean = false) {
    const config = loadOrInitConfig(workspaceFolder);
    const compilerPath = getCompilerPath(context, config);
    if (config.compilerPath && config.compilerPath.trim() !== '') {
        if (!validatePath(compilerPath)) {
            vscode.window.showErrorMessage(`JMCC: Путь к компилятору недействителен: ${compilerPath}`);
            return;
        }
    }
    const args = ['py', `"${compilerPath}"`, 'compile', `"${path.normalize(targetPath)}"`];
    if (mode === 'UPLOAD') {
        args.push('-u');
    } else if (mode === 'BOTH') {
        args.push('-su');
    }
    if ((mode === 'SAVE' || mode === 'BOTH' || isCompileAsFile) && config.compilerOutputPath && config.compilerOutputPath.trim() !== '') {
        const normalizedOutputPath = path.normalize(config.compilerOutputPath);
        if (!validatePath(normalizedOutputPath, true)) {
            vscode.window.showErrorMessage(`JMCC: Путь для вывода должен быть файлом: ${normalizedOutputPath}`);
            return;
        }
        args.push('-o', `"${normalizedOutputPath}"`);
    }
    runCommand(args.join(' '), workspaceFolder);
}

function decompile(targetPath: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext) {
    const config = loadOrInitConfig(workspaceFolder);
    const compilerPath = getCompilerPath(context, config);
    if (config.compilerPath && config.compilerPath.trim() !== '') {
        if (!validatePath(compilerPath)) {
            vscode.window.showErrorMessage(`JMCC: Путь к компилятору недействителен: ${compilerPath}`);
            return;
        }
    }
    const args = ['py', `"${compilerPath}"`, 'decompile', `"${path.normalize(targetPath)}"`];
    runCommand(args.join(' '), workspaceFolder);
}

function saveAndUpload(targetPath: string, workspaceFolder: vscode.WorkspaceFolder, context: vscode.ExtensionContext) {
    const config = loadOrInitConfig(workspaceFolder);
    const compilerPath = getCompilerPath(context, config);
    if (config.compilerPath && config.compilerPath.trim() !== '') {
        if (!validatePath(compilerPath)) {
            vscode.window.showErrorMessage(`JMCC: Путь к компилятору недействителен: ${compilerPath}`);
            return;
        }
    }
    const args = ['py', `"${compilerPath}"`, 'compile', `"${path.normalize(targetPath)}"`, '-su'];
    if (config.compilerOutputPath && config.compilerOutputPath.trim() !== '') {
        const normalizedOutputPath = path.normalize(config.compilerOutputPath);
        if (!validatePath(normalizedOutputPath, true)) {
            vscode.window.showErrorMessage(`JMCC: Путь для вывода должен быть файлом: ${normalizedOutputPath}`);
            return;
        }
        args.push('-o', `"${normalizedOutputPath}"`);
    }
    runCommand(args.join(' '), workspaceFolder);
}

export async function activate(context: vscode.ExtensionContext) {
    try {
        await checkAndUpdateAssets(context);
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
        vscode.window.showErrorMessage(`JMCC: Ошибка запуска LSP-сервера или проверки обновлений: ${err}`);
    }

    const existingTerminal = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
    if (existingTerminal) {
        existingTerminal.dispose();
    }

    const commands = [
        vscode.commands.registerCommand('jmcc.compileAsFile', (uri: vscode.Uri) => {
            const folder = getWorkspaceFolder(uri);
            if (folder) compile(uri.fsPath, 'SAVE', folder, context, true);
        }),
        vscode.commands.registerCommand('jmcc.compileAsUrl', (uri: vscode.Uri) => {
            const folder = getWorkspaceFolder(uri);
            if (folder) compile(uri.fsPath, 'UPLOAD', folder, context);
        }),
        vscode.commands.registerCommand('jmcc.decompileFile', (uri: vscode.Uri) => {
            const folder = getWorkspaceFolder(uri);
            if (folder) decompile(uri.fsPath, folder, context);
        }),
        vscode.commands.registerCommand('jmcc.compileActiveFile', () => {
            const active = vscode.window.activeTextEditor?.document;
            if (!active) return;
            if (active.isDirty) active.save();
            const folder = getWorkspaceFolder(active.uri);
            if (folder && active.fileName.endsWith('.jc')) {
                compile(active.fileName, 'SAVE', folder, context);
            }
        }),
        vscode.commands.registerCommand('jmcc.runActiveFile', () => {
            const active = vscode.window.activeTextEditor?.document;
            if (!active) return;
            if (active.isDirty) active.save();
            const filePath = active.fileName;
            const folder = getWorkspaceFolder(active.uri);
            if (!folder) return;
            const config = loadOrInitConfig(folder);
            if (filePath.endsWith('.jc')) {
                const mode = config.defaultCompileActiveFileMode?.toUpperCase() || 'UPLOAD';
                compile(filePath, mode, folder, context);
            } else if (filePath.endsWith('.json')) {
                decompile(filePath, folder, context);
            }
        }),
        vscode.commands.registerCommand('jmcc.saveAndUpload', (uri: vscode.Uri) => {
            const folder = getWorkspaceFolder(uri);
            if (folder) {
                const active = vscode.window.activeTextEditor?.document;
                if (active && active.uri.fsPath === uri.fsPath && active.isDirty) {
                    active.save();
                }
                saveAndUpload(uri.fsPath, folder, context);
            }
        })
    ];
    context.subscriptions.push(...commands);
}

export async function deactivate(): Promise<void> {
    if (client) {
        await client.stop();
        client = undefined;
    }
}