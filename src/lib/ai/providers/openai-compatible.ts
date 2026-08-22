import type {
  AIMessage,
  AIProvider,
  AIStreamChunk,
  AIToolCall,
  CompleteTextParams,
  StreamTextParams,
} from "@/types";

export interface OpenAICompatibleProviderOptions {
  apiKey: string;
  baseUrl?: string;
  id?: string;
}

type OpenAIChatMessage = {
  role: string;
  content: string | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
};

function toOpenAIMessages(messages: AIMessage[]): OpenAIChatMessage[] {
  return messages.map((m) => {
    const base: OpenAIChatMessage = {
      role: m.role,
      content: m.content,
    };
    if (m.name) base.name = m.name;
    if (m.toolCallId) base.tool_call_id = m.toolCallId;
    if (m.toolCalls?.length) {
      base.tool_calls = m.toolCalls.map((tc) => ({
        id: tc.id,
        type: "function" as const,
        function: { name: tc.name, arguments: tc.arguments },
      }));
    }
    return base;
  });
}

async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        break;
      }
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || line.startsWith(":")) continue;
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        try {
          yield JSON.parse(data) as unknown;
        } catch {
          // skip malformed SSE payloads
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export class OpenAICompatibleProvider implements AIProvider {
  readonly id: string;
  private readonly apiKey: string;
  private readonly baseUrl: string;

  constructor(options: OpenAICompatibleProviderOptions) {
    this.id = options.id ?? "openai-compatible";
    this.apiKey = options.apiKey;
    this.baseUrl = (options.baseUrl ?? "https://api.openai.com/v1").replace(
      /\/$/,
      "",
    );
  }

  async *streamText(params: StreamTextParams): AsyncIterable<AIStreamChunk> {
    const url = `${this.baseUrl}/chat/completions`;
    let response: Response;

    try {
      response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: params.model,
          messages: toOpenAIMessages(params.messages),
          stream: true,
          stream_options: { include_usage: true },
          temperature: params.temperature,
          max_tokens: params.maxTokens,
          tools: params.tools,
        }),
        signal: params.signal,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to reach model provider";
      yield { kind: "error", code: "PROVIDER_NETWORK_ERROR", message };
      return;
    }

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      yield {
        kind: "error",
        code: "PROVIDER_HTTP_ERROR",
        message: `Provider returned ${response.status}: ${text.slice(0, 500)}`,
      };
      return;
    }

    if (!response.body) {
      yield {
        kind: "error",
        code: "PROVIDER_EMPTY_BODY",
        message: "Provider response had no body",
      };
      return;
    }

    const pendingTools = new Map<
      number,
      { id: string; name: string; arguments: string }
    >();
    let finishReason: string | undefined;
    let usage:
      | { inputTokens?: number; outputTokens?: number }
      | undefined;

    for await (const chunk of parseSseStream(response.body, params.signal)) {
      const data = chunk as {
        choices?: Array<{
          delta?: {
            content?: string | null;
            tool_calls?: Array<{
              index?: number;
              id?: string;
              function?: { name?: string; arguments?: string };
            }>;
          };
          finish_reason?: string | null;
        }>;
        usage?: {
          prompt_tokens?: number;
          completion_tokens?: number;
        };
      };

      if (data.usage) {
        usage = {
          inputTokens: data.usage.prompt_tokens,
          outputTokens: data.usage.completion_tokens,
        };
      }

      const choice = data.choices?.[0];
      if (!choice) continue;

      if (choice.finish_reason) {
        finishReason = choice.finish_reason;
      }

      const delta = choice.delta;
      if (!delta) continue;

      if (typeof delta.content === "string" && delta.content.length > 0) {
        yield { kind: "content_delta", delta: delta.content };
      }

      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const index = tc.index ?? 0;
          const existing = pendingTools.get(index);
          if (!existing) {
            pendingTools.set(index, {
              id: tc.id ?? `tool_${index}`,
              name: tc.function?.name ?? "",
              arguments: tc.function?.arguments ?? "",
            });
          } else {
            if (tc.id) existing.id = tc.id;
            if (tc.function?.name) existing.name = tc.function.name;
            if (tc.function?.arguments) {
              existing.arguments += tc.function.arguments;
            }
          }
        }
      }
    }

    for (const tool of pendingTools.values()) {
      if (!tool.name) continue;
      yield {
        kind: "tool_start",
        toolCallId: tool.id,
        name: tool.name,
        arguments: tool.arguments || undefined,
      };
    }

    yield {
      kind: "message_complete",
      finishReason,
      usage,
    };
  }

  async completeText(params: CompleteTextParams): Promise<string> {
    const url = `${this.baseUrl}/chat/completions`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: params.model,
        messages: toOpenAIMessages(params.messages),
        stream: false,
        temperature: params.temperature ?? 0.3,
        max_tokens: params.maxTokens ?? 64,
      }),
      signal: params.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(
        `Provider completion failed (${response.status}): ${text.slice(0, 300)}`,
      );
    }

    const json = (await response.json()) as {
      choices?: Array<{ message?: { content?: string | null } }>;
    };
    return json.choices?.[0]?.message?.content?.trim() ?? "";
  }
}

/** Collect completed tool calls from a stream of tool_start events (last wins per id). */
export function collectToolCallsFromStarts(
  starts: Array<{ toolCallId: string; name: string; arguments?: string }>,
): AIToolCall[] {
  const map = new Map<string, AIToolCall>();
  for (const s of starts) {
    map.set(s.toolCallId, {
      id: s.toolCallId,
      name: s.name,
      arguments: s.arguments ?? "",
    });
  }
  return [...map.values()];
}
