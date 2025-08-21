# app/dash_assistant/ingestion/md_loader.py
"""Markdown loader for dashboard documentation."""
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.config import logger


async def load_markdown_dir(md_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load markdown files from directory.
    
    Args:
        md_dir: Directory containing markdown files
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping slug to content data
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")
    
    if not md_dir.is_dir():
        raise ValueError(f"Path is not a directory: {md_dir}")
    
    markdown_content = {}
    
    # Find all markdown files
    md_files = list(md_dir.glob("*.md"))
    
    for md_file in md_files:
        try:
            # Extract slug from filename (remove .md extension)
            slug = md_file.stem
            
            # Load and parse markdown content
            content_data = await load_markdown_file(md_file)
            markdown_content[slug] = content_data
            
        except Exception as e:
            logger.error(f"Error loading markdown file {md_file}: {e}")
            # Continue with other files
            continue
    
    logger.info(f"Loaded {len(markdown_content)} markdown files from {md_dir}")
    return markdown_content


async def load_markdown_file(md_file: Path) -> Dict[str, Any]:
    """Load and parse a single markdown file.
    
    Args:
        md_file: Path to markdown file
        
    Returns:
        Dict[str, Any]: Parsed content data
    """
    try:
        with open(md_file, 'r', encoding='utf-8') as file:
            raw_content = file.read()
        
        # Parse markdown content
        parsed_content = parse_markdown_content(raw_content)
        
        # Add metadata
        parsed_content['filename'] = md_file.name
        parsed_content['slug'] = md_file.stem
        
        return parsed_content
        
    except Exception as e:
        logger.error(f"Error reading markdown file {md_file}: {e}")
        raise


def parse_markdown_content(content: str) -> Dict[str, Any]:
    """Parse markdown content and extract structured information.
    
    Args:
        content: Raw markdown content
        
    Returns:
        Dict[str, Any]: Parsed content structure
    """
    result = {
        'content': content,
        'title': None,
        'sections': [],
        'headings': [],
        'links': [],
        'code_blocks': [],
        'metadata': {}
    }
    
    # Extract title (first H1 heading)
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        result['title'] = title_match.group(1).strip()
    
    # Extract all headings
    heading_pattern = r'^(#{1,6})\s+(.+)$'
    headings = re.findall(heading_pattern, content, re.MULTILINE)
    result['headings'] = [
        {'level': len(hashes), 'text': text.strip()}
        for hashes, text in headings
    ]
    
    # Extract sections based on headings
    sections = split_into_sections(content)
    result['sections'] = sections
    
    # Extract links
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.findall(link_pattern, content)
    result['links'] = [
        {'text': text, 'url': url}
        for text, url in links
    ]
    
    # Extract code blocks
    code_block_pattern = r'```(\w*)\n(.*?)\n```'
    code_blocks = re.findall(code_block_pattern, content, re.DOTALL)
    result['code_blocks'] = [
        {'language': lang or 'text', 'code': code.strip()}
        for lang, code in code_blocks
    ]
    
    return result


def split_into_sections(content: str) -> List[Dict[str, Any]]:
    """Split markdown content into sections based on headings.
    
    Args:
        content: Markdown content
        
    Returns:
        List[Dict[str, Any]]: List of sections
    """
    sections = []
    lines = content.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        # Check if line is a heading
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if heading_match:
            # Save previous section if exists
            if current_section:
                current_section['content'] = '\n'.join(current_content).strip()
                sections.append(current_section)
            
            # Start new section
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            
            current_section = {
                'level': level,
                'title': title,
                'content': ''
            }
            current_content = []
        else:
            # Add line to current section content
            if current_section:
                current_content.append(line)
    
    # Add last section
    if current_section:
        current_section['content'] = '\n'.join(current_content).strip()
        sections.append(current_section)
    
    return sections


def extract_key_phrases(content: str) -> List[str]:
    """Extract key phrases from markdown content for indexing.
    
    Args:
        content: Markdown content
        
    Returns:
        List[str]: List of key phrases
    """
    # Remove markdown formatting
    clean_content = remove_markdown_formatting(content)
    
    # Extract phrases using simple heuristics
    phrases = []
    
    # Extract phrases in bold/italic
    bold_italic_pattern = r'\*\*([^*]+)\*\*|\*([^*]+)\*|__([^_]+)__|_([^_]+)_'
    matches = re.findall(bold_italic_pattern, content)
    for match in matches:
        phrase = next(group for group in match if group)
        if phrase and len(phrase.strip()) > 2:
            phrases.append(phrase.strip())
    
    # Extract phrases from headings
    heading_pattern = r'^#{1,6}\s+(.+)$'
    headings = re.findall(heading_pattern, content, re.MULTILINE)
    phrases.extend([h.strip() for h in headings])
    
    # Extract quoted phrases
    quote_pattern = r'"([^"]+)"'
    quotes = re.findall(quote_pattern, clean_content)
    phrases.extend([q.strip() for q in quotes if len(q.strip()) > 2])
    
    # Remove duplicates and clean
    unique_phrases = []
    seen = set()
    for phrase in phrases:
        clean_phrase = phrase.lower().strip()
        if clean_phrase not in seen and len(clean_phrase) > 2:
            seen.add(clean_phrase)
            unique_phrases.append(phrase.strip())
    
    return unique_phrases


def remove_markdown_formatting(content: str) -> str:
    """Remove markdown formatting from content.
    
    Args:
        content: Markdown content
        
    Returns:
        str: Plain text content
    """
    # Remove code blocks
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    
    # Remove inline code
    content = re.sub(r'`([^`]+)`', r'\1', content)
    
    # Remove links but keep text
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    
    # Remove images
    content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', content)
    
    # Remove bold/italic formatting
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    content = re.sub(r'\*([^*]+)\*', r'\1', content)
    content = re.sub(r'__([^_]+)__', r'\1', content)
    content = re.sub(r'_([^_]+)_', r'\1', content)
    
    # Remove headings formatting
    content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
    
    # Remove horizontal rules
    content = re.sub(r'^---+$', '', content, flags=re.MULTILINE)
    
    # Remove blockquotes
    content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = re.sub(r'[ \t]+', ' ', content)
    
    return content.strip()


def create_content_chunks(content_data: Dict[str, Any], max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """Create content chunks from markdown data.
    
    Args:
        content_data: Parsed markdown content data
        max_chunk_size: Maximum size of each chunk
        
    Returns:
        List[Dict[str, Any]]: List of content chunks
    """
    chunks = []
    
    # Create chunk from title if available
    if content_data.get('title'):
        chunks.append({
            'scope': 'title',
            'content': content_data['title'],
            'lang': 'en'
        })
    
    # Create chunks from sections
    for section in content_data.get('sections', []):
        if section.get('content') and section['content'].strip():
            # Split large sections into smaller chunks
            section_chunks = split_text_into_chunks(
                section['content'], 
                max_chunk_size,
                section.get('title', '')
            )
            
            for chunk_content in section_chunks:
                chunks.append({
                    'scope': 'desc',
                    'content': chunk_content,
                    'lang': 'en',
                    'section_title': section.get('title')
                })
    
    # If no sections, create chunks from full content
    if not content_data.get('sections'):
        clean_content = remove_markdown_formatting(content_data['content'])
        if clean_content.strip():
            content_chunks = split_text_into_chunks(clean_content, max_chunk_size)
            for chunk_content in content_chunks:
                chunks.append({
                    'scope': 'desc',
                    'content': chunk_content,
                    'lang': 'en'
                })
    
    return chunks


def split_text_into_chunks(text: str, max_size: int, prefix: str = '') -> List[str]:
    """Split text into chunks of maximum size.
    
    Args:
        text: Text to split
        max_size: Maximum chunk size
        prefix: Optional prefix for context
        
    Returns:
        List[str]: List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # If text is small enough, return as single chunk
    if len(text) <= max_size:
        return [text.strip()]
    
    chunks = []
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = prefix + '\n\n' if prefix else ''
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If adding this paragraph would exceed max size
        if len(current_chunk) + len(paragraph) + 2 > max_size:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk
            if len(paragraph) > max_size:
                # Split large paragraph by sentences
                sentences = re.split(r'[.!?]+\s+', paragraph)
                current_chunk = prefix + '\n\n' if prefix else ''
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 > max_size:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                    else:
                        current_chunk += sentence + '. '
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk.strip():
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


class MarkdownLoadError(Exception):
    """Custom exception for markdown loading errors."""
    pass


async def load_markdown_for_slug(md_dir: Path, slug: str) -> Optional[Dict[str, Any]]:
    """Load markdown content for a specific dashboard slug.
    
    Args:
        md_dir: Directory containing markdown files
        slug: Dashboard slug to load
        
    Returns:
        Optional[Dict[str, Any]]: Content data or None if not found
    """
    md_file = md_dir / f"{slug}.md"
    
    if not md_file.exists():
        logger.debug(f"Markdown file not found for slug '{slug}': {md_file}")
        return None
    
    try:
        return await load_markdown_file(md_file)
    except Exception as e:
        logger.error(f"Error loading markdown for slug '{slug}': {e}")
        return None
