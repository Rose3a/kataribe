// ZIP 直下の kataribe.exe。初回セットアップ（tools\setup.ps1）を済ませてから editor\kataribe.exe を開く。
// tools\build_portable_zip.ps1 が csc（/target:winexe）でビルドする。
// まず窓を出さずに setup.ps1 -Quick（済んだセットアップの確認だけ）を動かし、
// 初回・フォルダの移動・追加の導入などで何か入れる必要があるときだけ、進み具合の見える窓で動かし直す。
using System;
using System.IO;
using System.Diagnostics;
using System.Threading;
using System.Windows.Forms;
class Launcher {
 const string Title = "kataribe";
 static int RunSetup(string box, bool visible) {
  var setup = new ProcessStartInfo("powershell.exe", "-NoProfile -ExecutionPolicy Bypass -File \"" + Path.Combine(box,"tools","setup.ps1") + "\"" + (visible ? "" : " -Quick"));
  setup.UseShellExecute=false; setup.WorkingDirectory=box; setup.CreateNoWindow=!visible;
  using (var process = Process.Start(setup)) { process.WaitForExit(); return process.ExitCode; }
 }
 [STAThread]
 static int Main(string[] args) {
  string box = AppDomain.CurrentDomain.BaseDirectory;
  bool acquired = false;
  using (var mutex = new Mutex(false, "Local\\IrodoriBox-" + box.ToLowerInvariant().GetHashCode().ToString("X"))) {
   try {
    try { acquired = mutex.WaitOne(0); } catch (AbandonedMutexException) { acquired = true; }
    if (!acquired) return 0;
    if (!File.Exists(Path.Combine(box,"tools","setup.ps1"))) {
     MessageBox.Show("tools\\setup.ps1 が見つかりません。ZIP をすべて展開したフォルダの kataribe.exe から起動してください。", Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
     return 1;
    }
    bool ready = File.Exists(Path.Combine(box,".local","setup.json")) && File.Exists(Path.Combine(box,".local","venv","Scripts","python.exe"));
    int code = ready ? RunSetup(box, false) : 1;
    if (code != 0) code = RunSetup(box, true);
    if (code != 0) {
     MessageBox.Show("セットアップに失敗しました。\n\n詳しくは logs\\first-setup.log を確認してください。setup.bat を実行すると、もう一度セットアップできます。", Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
     return 1;
    }
    string editor = null;
    foreach (var candidate in new[] {
     Path.Combine(box,"editor","kataribe.exe"),
     Path.Combine(box,"voicevox-editor","dist_electron","win-unpacked","kataribe.exe")
    }) if (File.Exists(candidate)) { editor = candidate; break; }
    ProcessStartInfo start;
    if (editor != null) {
     start = new ProcessStartInfo(editor);
     start.WorkingDirectory = Path.GetDirectoryName(editor); start.UseShellExecute=false;
    } else {
     string browser = Path.Combine(box,"open_browser.bat");
     if (!File.Exists(browser)) throw new FileNotFoundException("editor\\kataribe.exe が見つかりません", browser);
     start = new ProcessStartInfo(browser);
     start.WorkingDirectory = box; start.UseShellExecute=true;
    }
    using (var process = Process.Start(start)) { }
    return 0;
   } catch (Exception ex) {
    MessageBox.Show("起動できませんでした。\n\n" + ex.Message, Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
    return 1;
   }
   finally { if(acquired) mutex.ReleaseMutex(); }
  }
 }
}
