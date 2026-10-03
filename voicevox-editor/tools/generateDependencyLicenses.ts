// pnpm's inventory follows its symlinked dependency graph, including transitive
// dependencies. license-checker's root scan only found the 27 direct packages.
import { execFileSync } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

type PackageInfo = {
  name: string;
  paths: string[];
  license: string;
  homepage?: string;
};
const pnpm = process.env.npm_execpath;
if (!pnpm || !pnpm.includes("pnpm")) {
  throw new Error("Run this generator with pnpm run license:dependencies");
}
const inventory = JSON.parse(
  execFileSync(
    process.execPath,
    [pnpm, "licenses", "list", "--prod", "--json"],
    {
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024,
    },
  ),
) as Record<string, PackageInfo[]>;

async function noticeFiles(root: string, depth = 0): Promise<string[]> {
  const found: string[] = [];
  for (const item of await fs.readdir(root, { withFileTypes: true })) {
    const file = path.join(root, item.name);
    if (
      item.isFile() &&
      /^(licen[cs]e|copying|copyright|notice|third.?party)/i.test(item.name)
    ) {
      found.push(file);
    } else if (
      item.isDirectory() &&
      depth < 2 &&
      item.name !== "node_modules" &&
      !item.name.startsWith(".")
    ) {
      found.push(...(await noticeFiles(file, depth + 1)));
    }
  }
  return found.sort();
}
const licenses = [];
const snapshots: Record<string, string> = {
  "@pixi/colord": "pixi-colord",
  "@vue/devtools-api": "vue-devtools",
  isarray: "isarray",
  "utaformatix-data": "utaformatix-data",
};
for (const pkg of Object.values(inventory)
  .flat()
  .sort((a, b) => a.name.localeCompare(b.name, "en"))) {
  for (const root of pkg.paths) {
    const metadata = JSON.parse(
      await fs.readFile(path.join(root, "package.json"), "utf8"),
    ) as { version: string };
    const files = await noticeFiles(root);
    const snapshot = snapshots[pkg.name];
    if (!files.length && !snapshot)
      throw new Error(`Missing license text: ${pkg.name}`);
    const text = files.length
      ? (
          await Promise.all(
            files.map(
              async (file) =>
                `${path.relative(root, file)}\n\n${await fs.readFile(file, "utf8")}`,
            ),
          )
        ).join("\n\n--------------------\n\n")
      : await fs.readFile(
          path.resolve("../licenses", snapshot + ".txt"),
          "utf8",
        );
    licenses.push({
      name: pkg.name,
      version: metadata.version,
      license: pkg.license,
      ...(pkg.homepage?.startsWith("https://") ? { url: pkg.homepage } : {}),
      text,
    });
  }
}
const sevenZipRoot = path.resolve("vendored/7z");
licenses.push({
  name: "7-Zip",
  version: "",
  license: "LGPL-2.1-or-later / BSD（詳細は同梱本文）",
  url: "https://www.7-zip.org/license.txt",
  text: await fs.readFile(path.join(sevenZipRoot, "License.txt"), "utf8"),
});
const electronRoot = path.resolve("node_modules/electron");
const electron = JSON.parse(
  await fs.readFile(path.join(electronRoot, "package.json"), "utf8"),
) as { version: string };
licenses.push({
  name: "Electron",
  version: electron.version,
  license: "MIT (Chromium and other components: separate licenses)",
  url:
    "https://github.com/electron/electron/blob/v" +
    electron.version +
    "/LICENSE",
  text:
    (await fs.readFile(path.join(electronRoot, "dist/LICENSE"), "utf8")) +
    "\n\nChromium 等の第三者ライセンス全文は Electron 配布物に付属する LICENSES.chromium.html を参照してください。",
});
// 同じバージョンでもパッケージの LICENSE が CRLF のことがあり、PC によって
// 出力が変わると追跡中のファイルが書き換わる。本文の改行は LF にそろえる。
for (const license of licenses) {
  license.text = license.text.replace(/\r\n?/g, "\n");
}
await fs.writeFile(
  "public/dependency-licenses.json",
  JSON.stringify(licenses, null, 2) + "\n",
);
console.log(
  `Collected ${licenses.length} frontend / Electron dependency notices.`,
);
