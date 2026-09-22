import { expect, test } from "vitest";
import {
  getDefaultHotkeySettings,
  hotkeyActionNameSchema,
} from "@/domain/hotkeyAction";

test("すべてのホットキーに初期値が設定されている", async () => {
  const defaultHotkeySettings = getDefaultHotkeySettings({ isMac: false });
  const allActionNames = new Set(hotkeyActionNameSchema.options);
  const defaultHotkeyActionsNames = new Set(
    defaultHotkeySettings.map((setting) => setting.action),
  );
  expect(allActionNames).toEqual(defaultHotkeyActionsNames);
});

test("選択中のセリフを生成して再生するショートカットはCtrl+Enter", () => {
  const hotkey = getDefaultHotkeySettings({ isMac: false }).find(
    (setting) => setting.action === "選択中のセリフを生成して再生",
  );

  expect(hotkey?.combination).toBe("Ctrl Enter");
});
