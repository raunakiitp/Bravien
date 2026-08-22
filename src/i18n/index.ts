import { en } from "@/i18n/en";

type Leaves<T, P extends string = ""> = T extends object
  ? {
      [K in keyof T & string]: Leaves<
        T[K],
        P extends "" ? K : `${P}.${K}`
      >;
    }[keyof T & string]
  : P;

export type TranslationKey = Leaves<typeof en>;

function getByPath(obj: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((acc, part) => {
    if (acc && typeof acc === "object" && part in (acc as object)) {
      return (acc as Record<string, unknown>)[part];
    }
    return undefined;
  }, obj);
}

/**
 * Look up an English UI string by dotted key, e.g. `t("chat.placeholder")`.
 * Optional `{name}` interpolation via vars.
 */
export function t(
  key: TranslationKey | string,
  vars?: Record<string, string | number>,
): string {
  const value = getByPath(en, key);
  let str = typeof value === "string" ? value : key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      str = str.replaceAll(`{${k}}`, String(v));
    }
  }
  return str;
}

export { en };
