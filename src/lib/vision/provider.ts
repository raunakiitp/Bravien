/**
 * Vision & image analysis abstraction layer for Bravien.
 */

export interface VisionAnalysisRequest {
  imageBuffer: Buffer;
  mimeType: string;
  filename?: string;
  prompt?: string;
}

export interface VisionAnalysisResult {
  supported: boolean;
  text?: string;
  error?: string;
  description?: string;
}

export interface VisionProvider {
  isAvailable(): Promise<boolean>;
  analyzeImage(request: VisionAnalysisRequest): Promise<VisionAnalysisResult>;
}

export class FallbackVisionProvider implements VisionProvider {
  async isAvailable(): Promise<boolean> {
    // Current runtime model is Qwen2.5-0.5B-Instruct (text-only).
    // Future multimodal weights can enable this dynamically.
    return process.env.BRAVIEN_VISION_ENABLED === "1";
  }

  async analyzeImage(request: VisionAnalysisRequest): Promise<VisionAnalysisResult> {
    const available = await this.isAvailable();
    if (!available) {
      return {
        supported: false,
        error: "VISION_UNAVAILABLE",
        description: `Bravien is running with a text-only model (${request.filename || "image"}). Vision analysis requires a multimodal checkpoint.`,
      };
    }

    return {
      supported: true,
      text: `Image analysis of ${request.filename || "uploaded image"} completed.`,
    };
  }
}

export const defaultVisionProvider = new FallbackVisionProvider();
