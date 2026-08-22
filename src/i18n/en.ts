export const en = {
  appName: "Bravien",
  appTagline: "Your AI assistant",

  nav: {
    chat: "Chat",
    settings: "Settings",
    memories: "Memories",
    signIn: "Sign in",
    signOut: "Sign out",
    newChat: "New chat",
  },

  chat: {
    placeholder: "Message Bravien…",
    send: "Send",
    stop: "Stop",
    thinking: "Thinking…",
    emptyTitle: "How can I help?",
    emptySubtitle: "Ask anything — write, code, reason, or explore.",
    copy: "Copy",
    regenerate: "Regenerate",
    like: "Good response",
    dislike: "Bad response",
  },

  models: {
    label: "Model",
    fast: "Fast",
    balanced: "Balanced",
    reasoning: "Reasoning",
    vision: "Vision",
  },

  errors: {
    generic: "Something went wrong. Please try again.",
    unauthorized: "Please sign in to continue.",
    rateLimited: "Too many requests. Slow down and try again.",
    modelUnavailable: "This model is unavailable right now.",
    uploadFailed: "Upload failed.",
    fileTooLarge: "File is too large.",
    unsupportedFile: "Unsupported file type.",
  },

  settings: {
    title: "Settings",
    profile: "Profile",
    preferences: "Preferences",
    memories: "Memories",
    appearance: "Appearance",
    save: "Save changes",
  },

  auth: {
    email: "Email",
    password: "Password",
    signIn: "Sign in",
    signUp: "Create account",
    continueWith: "Continue with {provider}",
  },
} as const;

export type EnStrings = typeof en;
