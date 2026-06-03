const vscode = require("vscode");
const path = require("path");
const { spawn } = require("child_process");
const readline = require("readline");

let proc, rl, pending = new Map(), reqId = 0;

function serverScript() {
  return "D:\\VS\\P3\\rnn-keras-autocomplete\\server_stdio.py";
}

function request(method, fields) {
  return new Promise((resolve, reject) => {
    if (!proc) {
      const py = "py";
      const script = serverScript();
      proc = spawn(py, [script], { cwd: path.dirname(script), stdio: ["pipe", "pipe", "pipe"] });
      rl = readline.createInterface({ input: proc.stdout });
      rl.on("line", (line) => {
        const msg = JSON.parse(line);
        if (pending.has(msg._id)) {
          pending.get(msg._id)(msg);
          pending.delete(msg._id);
        }
      });
    }
    const id = ++reqId;
    const timer = setTimeout(() => reject(new Error("timeout")), 120000);
    pending.set(id, (msg) => {
      clearTimeout(timer);
      msg.ok ? resolve(msg) : reject(new Error(msg.error || "error"));
    });
    proc.stdin.write(JSON.stringify({ method, _id: id, ...fields }) + "\n");
  });
}

async function completeLine() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return;
  const pos = ed.selection.active;
  const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);
  const maxNew = vscode.workspace.getConfiguration("rnnKeras").get("maxNew") || 360;
  const res = await request("complete", { prefix, max_new: maxNew, temperature: 0.50 });
  const suffix = res.text.slice(prefix.length);
  await ed.edit((eb) => eb.insert(pos, suffix));
}

async function showSuggestions() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return;
  const pos = ed.selection.active;
  const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);
  const res = await request("suggest", { prefix, n: 5 });
  const pick = await vscode.window.showQuickPick(res.items, { placeHolder: "Sugerencias RNN" });
  if (!pick) return;
  await ed.edit((eb) => eb.insert(pos, pick.slice(prefix.length)));
}

function activate(ctx) {
  vscode.window.showInformationMessage("Extension RNN C cargada");

  ctx.subscriptions.push(
    vscode.commands.registerCommand("rnnKeras.complete", completeLine),
    vscode.commands.registerCommand("rnnKeras.suggest", showSuggestions)
  );
}

function deactivate() {
  if (proc) proc.kill();
}

module.exports = { activate, deactivate };