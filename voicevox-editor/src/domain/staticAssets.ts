import { z } from "zod";
import type { UpdateInfo } from "@/type/preload";

const ossLicenseInfoSchema = z.object({
  name: z.string(),
  version: z.string().nullable().optional(),
  license: z.string().nullable().optional(),
  url: z.url().optional(),
  text: z.string(),
});

export type OssLicenseInfo = z.infer<typeof ossLicenseInfoSchema>;

const loadDefault = async <T>(
  loader: () => Promise<{ default: T }>,
): Promise<T> => {
  const module = await loader();
  return module.default;
};

export const loadHowToUseText = async (): Promise<string> => {
  return await loadDefault(
    () => import("../../public/irodori-howtouse.md?raw"),
  );
};

export const loadContactText = async (): Promise<string> => {
  return await loadDefault(() => import("../../public/irodori-contact.md?raw"));
};

export const loadQAndAText = async (): Promise<string> => {
  return await loadDefault(() => import("../../public/irodori-qAndA.md?raw"));
};

export const loadPolicyText = async (): Promise<string> => {
  return await loadDefault(() => import("../../public/policy.md?raw"));
};

// 実行環境（Python パッケージ）のライセンスは PC ごとに違うため、セットアップは
// 追跡しない runtime-licenses.local.json に書く。無いチェックアウトでは追跡中の
// runtime-licenses.json を使う。glob なのでファイルが無くてもビルドは通る。
const runtimeLicenseFiles = import.meta.glob<unknown[]>(
  [
    "../../public/runtime-licenses.json",
    "../../public/runtime-licenses.local.json",
  ],
  { import: "default" },
);

const loadRuntimeLicenses = async (): Promise<unknown[]> => {
  const load =
    runtimeLicenseFiles["../../public/runtime-licenses.local.json"] ??
    runtimeLicenseFiles["../../public/runtime-licenses.json"];
  return await load();
};

export const loadOssLicenses = async (): Promise<OssLicenseInfo[]> => {
  const groups = await Promise.all([
    loadDefault(() => import("../../public/licenses.json")),
    loadDefault(() => import("../../public/dependency-licenses.json")),
    loadRuntimeLicenses(),
  ]);
  return ossLicenseInfoSchema.array().parse(groups.flat());
};

export const loadUpdateInfos = async (): Promise<UpdateInfo[]> => {
  return await loadDefault(() => import("../../public/updateInfos.json"));
};

export const loadOssCommunityInfos = async (): Promise<string> => {
  return await loadDefault(
    () => import("../../public/irodori-community.md?raw"),
  );
};

export const loadPrivacyPolicyText = async (): Promise<string> => {
  return await loadDefault(() => import("../../public/privacyPolicy.md?raw"));
};
