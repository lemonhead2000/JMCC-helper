import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { LanguageClient, LanguageClientOptions, ServerOptions, Executable } from 'vscode-languageclient/node';

const TERMINAL_NAME = 'JMCC Terminal';

let client: LanguageClient | undefined;

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
    const outputChannel = vscode.window.createOutputChannel('JMCC Language Server Debug');

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

function compile(targetPath: string, mode: string, workspaceFolder: vscode.WorkspaceFolder, isCompileAsFile: boolean = false) {
    const config = loadOrInitConfig(workspaceFolder);
    if (!config.compilerPath) {
        vscode.window.showErrorMessage('JMCC: Укажите путь к компилятору Python в .jmccconfig.json');
        return;
    }

    const normalizedCompilerPath = path.normalize(config.compilerPath);
    if (!validatePath(normalizedCompilerPath)) {
        vscode.window.showErrorMessage(`JMCC: Путь к компилятору недействителен: ${normalizedCompilerPath}`);
        return;
    }

    const args = ['py', `"${normalizedCompilerPath}"`, 'compile', `"${path.normalize(targetPath)}"`];
    
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

function decompile(targetPath: string, workspaceFolder: vscode.WorkspaceFolder) {
    const config = loadOrInitConfig(workspaceFolder);
    if (!config.compilerPath) {
        vscode.window.showErrorMessage('JMCC: Укажите путь к компилятору Python в .jmccconfig.json');
        return;
    }

    const normalizedCompilerPath = path.normalize(config.compilerPath);
    if (!validatePath(normalizedCompilerPath)) {
        vscode.window.showErrorMessage(`JMCC: Путь к компилятору недействителен: ${normalizedCompilerPath}`);
        return;
    }

    const args = ['py', `"${normalizedCompilerPath}"`, 'decompile', `"${path.normalize(targetPath)}"`];
    runCommand(args.join(' '), workspaceFolder);
}

function saveAndUpload(targetPath: string, workspaceFolder: vscode.WorkspaceFolder) {
    const config = loadOrInitConfig(workspaceFolder);
    if (!config.compilerPath) {
        vscode.window.showErrorMessage('JMCC: Укажите путь к компилятору Python в .jmccconfig.json');
        return;
    }

    const normalizedCompilerPath = path.normalize(config.compilerPath);
    if (!validatePath(normalizedCompilerPath)) {
        vscode.window.showErrorMessage(`JMCC: Путь к компилятору недействителен: ${normalizedCompilerPath}`);
        return;
    }

    const args = ['py', `"${normalizedCompilerPath}"`, 'compile', `"${path.normalize(targetPath)}"`, '-su'];

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
        vscode.window.showErrorMessage(`JMCC: Ошибка запуска LSP-сервера: ${err}`);
    }

    const existingTerminal = vscode.window.terminals.find(t => t.name === TERMINAL_NAME);
    if (existingTerminal) {
        existingTerminal.dispose();
    }

    const commands = [
        vscode.commands.registerCommand('jmcc.compileAsFile', (uri: vscode.Uri) => {
            const folder = getWorkspaceFolder(uri);
            if (folder) compile(uri.fsPath, 'SAVE', folder, true);
        }),

        vscode.commands.registerCommand('jmcc.compileAsUrl', (uri: vscode.Uri) => {
            const folder = getWorkspaceFolder(uri);
            if (folder) compile(uri.fsPath, 'UPLOAD', folder);
        }),

        vscode.commands.registerCommand('jmcc.decompileFile', (uri: vscode.Uri) => {
            const folder = getWorkspaceFolder(uri);
            if (folder) decompile(uri.fsPath, folder);
        }),

        vscode.commands.registerCommand('jmcc.compileActiveFile', () => {
            const active = vscode.window.activeTextEditor?.document;
            if (!active) return;
            if (active.isDirty) active.save();
            const folder = getWorkspaceFolder(active.uri);
            if (folder && active.fileName.endsWith('.jc')) {
                compile(active.fileName, 'SAVE', folder);
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
                compile(filePath, mode, folder);
            } else if (filePath.endsWith('.json')) {
                decompile(filePath, folder);
            }
        }),

        vscode.commands.registerCommand('jmcc.saveAndUpload', (uri: vscode.Uri) => {
            const folder = getWorkspaceFolder(uri);
            if (folder) {
                const active = vscode.window.activeTextEditor?.document;
                if (active && active.uri.fsPath === uri.fsPath && active.isDirty) {
                    active.save();
                }
                saveAndUpload(uri.fsPath, folder);
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