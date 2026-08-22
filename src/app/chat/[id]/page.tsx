/** `/chat/[id]` — a saved conversation. */

import { ChatWorkspace } from "@/components/chat/workspace";

export const dynamic = "force-dynamic";

export default async function ChatPage(props: PageProps<"/chat/[id]">) {
  const { id } = await props.params;
  // The conversation itself is fetched client-side: the same component serves `/`,
  // and the runtime state it needs alongside is only observable from the browser.
  return <ChatWorkspace conversationId={id} />;
}
