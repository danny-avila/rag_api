import type { Chunk, Segment } from "./contracts";

function boundary(text: string, position: number): number {
  const code = text.charCodeAt(position - 1);
  return code >= 0xd800 && code <= 0xdbff ? position - 1 : position;
}

export function* chunks(
  segments: readonly Segment[],
  size = 1500,
  overlap = 150,
): Generator<Chunk> {
  if (size < 4 || overlap < 0 || overlap >= size - 2)
    throw new Error("INVALID_CHUNK_SIZE");
  let index = 0;
  for (const segment of segments) {
    const sections: Array<{ start: number; end: number; path: string[] }> = [];
    const stack: Array<{ depth: number; title: string }> = [];
    let start = 0;
    let path: string[] = [];
    let offset = 0;
    let fenced = false;
    for (const line of segment.text.split("\n")) {
      if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
      const heading = fenced ? null : /^(#{1,6})[ \t]+(.+?)\s*$/.exec(line);
      if (heading) {
        sections.push({ start, end: offset, path });
        const depth = heading[1]!.length;
        while (stack.length && stack.at(-1)!.depth >= depth) stack.pop();
        const title = heading[2]!;
        stack.push({
          depth,
          title: title.slice(0, boundary(title, Math.min(512, title.length))),
        });
        path = stack.map((entry) => entry.title);
        start = offset;
      }
      offset += line.length + 1;
    }
    sections.push({ start, end: segment.text.length, path });
    for (const section of sections) {
      let position = section.start;
      while (position < section.end) {
        const end = boundary(
          segment.text,
          Math.min(position + size, section.end),
        );
        const text = segment.text.slice(position, end);
        if (text.trim())
          yield {
            index: index++,
            text,
            page: segment.kind === "page" ? segment.index : null,
            segment: segment.index,
            start: position,
            end,
            section: section.path,
          };
        if (end >= section.end) break;
        position = Math.max(
          position + 1,
          boundary(segment.text, end - overlap),
        );
      }
    }
  }
}
