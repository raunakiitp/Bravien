/**
 * Browser-native voice / speech architecture for Bravien.
 * Uses Web Speech API (SpeechRecognition and SpeechSynthesis) without requiring external paid APIs.
 */

export interface SpeechInputOptions {
  onTranscript: (text: string, isFinal: boolean) => void;
  onError?: (error: string) => void;
  onEnd?: () => void;
  language?: string;
}

export interface SpeechOutputOptions {
  text: string;
  onStart?: () => void;
  onEnd?: () => void;
  onError?: (error: string) => void;
  rate?: number;
  pitch?: number;
  language?: string;
}

export function isSpeechRecognitionSupported(): boolean {
  if (typeof window === "undefined") return false;
  return Boolean(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).SpeechRecognition ||
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).webkitSpeechRecognition,
  );
}

export function isSpeechSynthesisSupported(): boolean {
  if (typeof window === "undefined") return false;
  return Boolean(window.speechSynthesis);
}

export class BrowserSpeechInputProvider {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private recognition: any = null;
  private isListening = false;

  start(options: SpeechInputOptions): boolean {
    if (!isSpeechRecognitionSupported()) {
      options.onError?.("Speech recognition is not supported in this browser.");
      return false;
    }

    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = options.language || "en-US";

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.recognition.onresult = (event: any) => {
        let finalTranscript = "";
        let interimTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        const text = finalTranscript || interimTranscript;
        if (text) {
          options.onTranscript(text, Boolean(finalTranscript));
        }
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.recognition.onerror = (event: any) => {
        options.onError?.(event.error || "Speech recognition error");
      };

      this.recognition.onend = () => {
        this.isListening = false;
        options.onEnd?.();
      };

      this.recognition.start();
      this.isListening = true;
      return true;
    } catch (err) {
      options.onError?.(err instanceof Error ? err.message : "Failed to start speech recognition");
      return false;
    }
  }

  stop(): void {
    if (this.recognition && this.isListening) {
      try {
        this.recognition.stop();
      } catch {
        // Ignore
      }
      this.isListening = false;
    }
  }

  get active(): boolean {
    return this.isListening;
  }
}

export class BrowserSpeechOutputProvider {
  speak(options: SpeechOutputOptions): boolean {
    if (!isSpeechSynthesisSupported()) {
      options.onError?.("Text-to-speech is not supported in this browser.");
      return false;
    }

    try {
      window.speechSynthesis.cancel(); // Stop any ongoing speech
      const utterance = new SpeechSynthesisUtterance(options.text);
      utterance.rate = options.rate ?? 1.0;
      utterance.pitch = options.pitch ?? 1.0;
      utterance.lang = options.language ?? "en-US";

      utterance.onstart = () => options.onStart?.();
      utterance.onend = () => options.onEnd?.();
      utterance.onerror = (e) => options.onError?.(e.error);

      window.speechSynthesis.speak(utterance);
      return true;
    } catch (err) {
      options.onError?.(err instanceof Error ? err.message : "Speech synthesis failed");
      return false;
    }
  }

  stop(): void {
    if (isSpeechSynthesisSupported()) {
      window.speechSynthesis.cancel();
    }
  }

  get speaking(): boolean {
    if (!isSpeechSynthesisSupported()) return false;
    return window.speechSynthesis.speaking;
  }
}
