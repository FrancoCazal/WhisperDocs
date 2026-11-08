// Loaded on demand by Markdown: highlight.js's grammars outweigh the rest of
// the chat, and most answers have no code.
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";

// Only fences that name their language are colored: guessing one ran every
// grammar over each unnamed block again on every streamed token. An unknown
// name leaves the block plain.
export const highlightPlugin = rehypeHighlight;
