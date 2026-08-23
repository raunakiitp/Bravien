/**
 * Web Research Provider Abstraction & SSRF Protection.
 */

import { logger } from "@/lib/observability/logger";

export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  domain: string;
  score: number;
  publishedDate?: string;
  content?: string;
}

export interface WebSearchOptions {
  maxResults?: number;
  timeoutMs?: number;
}

export interface WebFetchOptions {
  maxContentLength?: number;
  timeoutMs?: number;
}

export interface WebSearchProvider {
  name: string;
  search(query: string, options?: WebSearchOptions): Promise<WebSearchResult[]>;
  fetchPage(url: string, options?: WebFetchOptions): Promise<string | null>;
}

/**
 * SSRF & URL Safety Validation.
 * Rejects private networks, local IPs, cloud metadata IPs, and non-HTTP protocols.
 */
export function isSafeUrl(urlString: string): boolean {
  try {
    const parsed = new URL(urlString);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return false;
    }

    const host = parsed.hostname.toLowerCase().trim();

    // Block localhost, private domain names
    if (
      host === "localhost" ||
      host.endsWith(".local") ||
      host.endsWith(".internal") ||
      host.endsWith(".localhost")
    ) {
      return false;
    }

    // Check for IPv4 addresses
    const ipv4Match = host.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
    if (ipv4Match) {
      const [, a, b, c, d] = ipv4Match.map(Number);
      if (a < 0 || a > 255 || b < 0 || b > 255 || c < 0 || c > 255 || d < 0 || d > 255) {
        return false;
      }
      // 127.0.0.0/8 (Loopback)
      if (a === 127) return false;
      // 0.0.0.0/8
      if (a === 0) return false;
      // 10.0.0.0/8 (Private)
      if (a === 10) return false;
      // 172.16.0.0/12 (Private: 172.16.0.0 - 172.31.255.255)
      if (a === 172 && b >= 16 && b <= 31) return false;
      // 192.168.0.0/16 (Private)
      if (a === 192 && b === 168) return false;
      // 169.254.0.0/16 (Link-local & AWS/GCP Metadata 169.254.169.254)
      if (a === 169 && b === 254) return false;
      // 100.64.0.0/10 (Carrier-grade NAT)
      if (a === 100 && b >= 64 && b <= 127) return false;
    }

    // Check for IPv6 addresses
    if (host.startsWith("[") || host.includes(":")) {
      if (host === "::1" || host === "[::1]") return false;
      if (host.startsWith("fe80:") || host.startsWith("[fe80:")) return false; // link local
      if (host.startsWith("fc") || host.startsWith("fd")) return false; // unique local
    }

    return true;
  } catch {
    return false;
  }
}

/**
 * Strips HTML tags and script/style content into clean readable text.
 */
export function sanitizeHtmlToText(html: string, maxLength: number = 1500): string {
  if (!html) return "";
  const cleaned = html
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, " ")
    .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, " ")
    .replace(/<noscript\b[^<]*(?:(?!<\/noscript>)<[^<]*)*<\/noscript>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();

  return cleaned.slice(0, maxLength);
}

/**
 * Built-in Lightweight Web Search Provider.
 * Fetches public HTML search results with timeout, URL deduplication, and SSRF filtering.
 */
export class PublicWebSearchProvider implements WebSearchProvider {
  name = "public-web";

  async search(
    query: string,
    options: WebSearchOptions = {},
  ): Promise<WebSearchResult[]> {
    const maxResults = options.maxResults ?? 4;
    const timeoutMs = options.timeoutMs ?? 6000;
    const q = query.trim();
    if (!q) return [];

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      // Use DuckDuckGo HTML lite search endpoint
      const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`;
      const res = await fetch(searchUrl, {
        signal: controller.signal,
        headers: {
          "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          Accept: "text/html",
        },
      });

      clearTimeout(timer);
      if (!res.ok) {
        logger.warn("web_search.http_failed", { status: res.status });
        return this.getFallbackResults(q, maxResults);
      }

      const html = await res.text();
      return this.parseDuckDuckGoHtml(html, maxResults);
    } catch (err) {
      clearTimeout(timer);
      logger.info("web_search.network_unavailable", {
        message: err instanceof Error ? err.message : String(err),
      });
      return this.getFallbackResults(q, maxResults);
    }
  }

  async fetchPage(
    url: string,
    options: WebFetchOptions = {},
  ): Promise<string | null> {
    if (!isSafeUrl(url)) {
      logger.warn("web_fetch.ssrf_blocked", { url });
      return null;
    }

    const maxLen = options.maxContentLength ?? 2000;
    const timeoutMs = options.timeoutMs ?? 5000;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(url, {
        signal: controller.signal,
        headers: {
          "User-Agent": "BravienBot/1.0",
          Accept: "text/html,text/plain",
        },
      });
      clearTimeout(timer);
      if (!res.ok) return null;

      const html = await res.text();
      return sanitizeHtmlToText(html, maxLen);
    } catch {
      clearTimeout(timer);
      return null;
    }
  }

  private parseDuckDuckGoHtml(html: string, maxResults: number): WebSearchResult[] {
    const results: WebSearchResult[] = [];
    const seenUrls = new Set<string>();

    // Regex to match search result blocks: class="result__body"
    const resultRegex = /<a class="result__url"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a class="result__snippet"[^>]*>([\s\S]*?)<\/a>/gi;
    let match: RegExpExecArray | null;

    while ((match = resultRegex.exec(html)) !== null && results.length < maxResults) {
      let rawUrl = match[1];
      // DuckDuckGo redirects through /l/?uddg=...
      const uddgMatch = rawUrl.match(/uddg=([^&]+)/);
      if (uddgMatch) {
        try {
          rawUrl = decodeURIComponent(uddgMatch[1]);
        } catch {
          // ignore
        }
      }

      if (!isSafeUrl(rawUrl)) continue;

      let domain = "";
      try {
        domain = new URL(rawUrl).hostname;
      } catch {
        continue;
      }

      if (seenUrls.has(rawUrl)) continue;
      seenUrls.add(rawUrl);

      const title = sanitizeHtmlToText(match[2], 120) || domain;
      const snippet = sanitizeHtmlToText(match[3], 300);

      if (snippet.length > 10) {
        results.push({
          title,
          url: rawUrl,
          snippet,
          domain,
          score: 1.0 - results.length * 0.1,
        });
      }
    }

    return results;
  }

  private getFallbackResults(query: string, maxResults: number): WebSearchResult[] {
    return [
      {
        title: `Search reference: ${query}`,
        url: `https://duckduckgo.com/?q=${encodeURIComponent(query)}`,
        snippet: `Public web search query processed for "${query}". External live web network access is optional or running offline.`,
        domain: "duckduckgo.com",
        score: 0.8,
      },
    ].slice(0, maxResults);
  }
}

export const defaultWebSearchProvider: WebSearchProvider = new PublicWebSearchProvider();
