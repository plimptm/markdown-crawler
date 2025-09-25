from bs4 import BeautifulSoup
import urllib.parse
import threading
from markdownify import markdownify as md
import requests
import logging
import queue
import time
import os
import re
from typing import (
    List,
    Optional,
    Union
)
__version__ = '0.1'
__author__ = 'Paul Pierre (github.com/paulpierre)'
__copyright__ = "(C) 2023 Paul Pierre. MIT License."
__contributors__ = ['Paul Pierre']

BANNER = r"""
                |                                     |             
 __ `__ \    _` |        __|   __|   _` | \ \  \   /  |   _ \   __| 
 |   |   |  (   |       (     |     (   |  \ \  \ /   |   __/  |    
_|  _|  _| \__._|      \___| _|    \__._|   \_/\_/   _| \___| _|    

-------------------------------------------------------------------------
A multithreaded 🕸️ web crawler that recursively crawls a website and
creates a 🔽 markdown file for each page by https://github.com/paulpierre
-------------------------------------------------------------------------
"""

logger = logging.getLogger(__name__)
DEFAULT_BASE_DIR = 'markdown'
DEFAULT_MAX_DEPTH = 3
DEFAULT_NUM_THREADS = 5
DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']
DEFAULT_TARGET_LINKS = ['body']
DEFAULT_DOMAIN_MATCH = True
DEFAULT_BASE_PATH_MATCH = True


# ---------------------------------
# URL Queue with unique set handler
# ---------------------------------

class URLQueue(queue.Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize)
        self._complete = set()

    def screen(self, url):
        if url in self._complete:
            #logger.debug(f'Skipped {item[1]} (already handled in queue)')
            return False
        elif url.split('.')[-1] in ['pdf', 'docx', 'xlsx']:
            #logger.debug(f'Skipped {item[1]} (not text/html)')
            return False
        else:
            return True

    def put(self, item, block=True, timeout=None):
        """ expecting tuples of (depth, url) """
        if self.screen(item[1]):
            super().put(item, block=block, timeout=timeout)
            self._complete.add(item[1])
        else:
            pass

    def get(self, block=True, timeout=None):
        item = super().get(block=block, timeout=timeout)
        return item

# --------------
# URL validation
# --------------
def is_valid_url(url: str) -> bool:
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        logger.debug(f'❌ Invalid URL {url}')
        return False


# ----------------
# Clean up the URL
# ----------------
def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), None, None, None))


# ------------------
# HTML parsing logic
# ------------------
def crawl(
    url: str,
    base_url: str,
    already_crawled: set,
    file_path: str,
    target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
    target_content: Union[str, List[str]] = None,
    valid_paths: Union[str, List[str]] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH,
    is_links: Optional[bool] = False
) -> List[str]:

    try:
        logger.debug(f'Crawling: {url}')
        
        # Headers to mimic a real browser and avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Check domain before making request if domain matching is required
        if is_domain_match:
            url_domain = urllib.parse.urlparse(url).netloc
            base_domain = urllib.parse.urlparse(base_url).netloc
            if url_domain != base_domain:
                logger.debug(f'Skipping {url} - domain {url_domain} does not match base domain {base_domain}')
                return []
       
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=False)
        
        # Handle redirects manually to check domain on final URL
        if response.status_code in [301, 302, 303, 307, 308]:
            redirect_url = response.headers.get('Location')
            if redirect_url:
                # Make redirect URL absolute
                redirect_url = urllib.parse.urljoin(url, redirect_url)
                logger.debug(f'Redirect from {url} to {redirect_url}')
                
                # Check if redirect stays within allowed domain
                if is_domain_match:
                    redirect_domain = urllib.parse.urlparse(redirect_url).netloc
                    base_domain = urllib.parse.urlparse(base_url).netloc
                    if redirect_domain != base_domain:
                        logger.debug(f'Skipping redirect to {redirect_url} - domain {redirect_domain} does not match base domain {base_domain}')
                        return []
                
                # Follow the redirect
                response = requests.get(redirect_url, headers=headers, timeout=10)
                # Update url to the final redirected URL for further processing
                url = redirect_url
                
    except requests.exceptions.RequestException as e:
        logger.error(f'❌ Request error for {url}: {e}')
        return []
    
    # Check for non-successful status codes
    if response.status_code != 200:
        if response.status_code == 403:
            logger.error(f'❌ Access forbidden (403) for {url}')
            logger.debug(f'Response headers: {dict(response.headers)}')
            if response.text:
                logger.debug(f'Response body: {response.text[:500]}...' if len(response.text) > 500 else f'Response body: {response.text}')
        elif response.status_code == 404:
            logger.error(f'❌ Page not found (404) for {url}')
        elif response.status_code == 429:
            logger.error(f'❌ Rate limited (429) for {url}')
            logger.debug(f'Response headers: {dict(response.headers)}')
        else:
            logger.error(f'❌ HTTP {response.status_code} error for {url}')
            logger.debug(f'Response headers: {dict(response.headers)}')
            if response.text:
                logger.debug(f'Response body: {response.text[:500]}...' if len(response.text) > 500 else f'Response body: {response.text}')
        return []
    
    if 'text/html' not in response.headers.get('Content-Type', ''):
        logger.error(f'❌ Content not text/html for {url}')
        return []
    already_crawled.add(url)
    logger.debug(f'Crawled {len(already_crawled)} URLs so far')

    # ---------------------------------
    # List of elements we want to strip
    # ---------------------------------
    strip_elements = []

    if is_links:
        strip_elements = ['a']

    # -------------------------------
    # Create BS4 instance for parsing
    # -------------------------------
    soup = BeautifulSoup(response.text, 'html.parser')

    # Strip unwanted tags
    for script in soup(['script', 'style']):
        script.decompose()

    # --------------------------------------------
    # Write the markdown file if it does not exist
    # --------------------------------------------
    if not os.path.exists(file_path):

        file_name = file_path.split("/")[-1]

        # ------------------
        # Get target content
        # ------------------

        content = get_target_content(soup, target_content=target_content)

        if content:
            # --------------
            # Parse markdown
            # --------------
            output = md(
                content,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )

            logger.info(f'Created 📝 {file_name}')

            # ------------------------------
            # Write markdown content to file
            # ------------------------------
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(output)
        else:
            logger.error(f'❌ Empty content for {file_path}. Target selectors: {target_content}')
            logger.debug(f'Available elements on page: {[tag.name for tag in soup.find_all()][:20]}')  # Show first 20 element types

    child_urls = get_target_links(
        soup,
        base_url,
        target_links,
        valid_paths=valid_paths,
        is_domain_match=is_domain_match,
        is_base_path_match=is_base_path_match    
    )

    logger.debug(f'Found {len(child_urls) if child_urls else 0} child URLs')
    #move sleep here to allow faster queue handling
    time.sleep(1)
    return child_urls


def get_target_content(
    soup: BeautifulSoup,
    target_content: Union[List[str], None] = None
) -> str:

    content = ''

    # -------------------------------------
    # Get target content by target selector
    # -------------------------------------
    if target_content:
        for target in target_content:
            # Use CSS selector instead of find_all for more flexibility
            elements = soup.select(target)
            if elements:
                for element in elements:
                    content += f'{str(element)}'.replace('\n', '')
            else:
                logger.debug(f'No elements found for selector: {target}')

    # ---------------------------
    # Naive estimation of content
    # ---------------------------
    else:
        max_text_length = 0
        main_content = None
        for tag in soup.find_all(DEFAULT_TARGET_CONTENT):
            text_length = len(tag.get_text())
            if text_length > max_text_length:
                max_text_length = text_length
                main_content = tag

        if main_content:
            content = str(main_content)

    return content if len(content) > 0 else False


def get_target_links(
    soup: BeautifulSoup,
    base_url: str,
    target_links: List[str] = DEFAULT_TARGET_LINKS,
    valid_paths: Union[List[str], None] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH
) -> List[str]:

    child_urls = []

    # Get all urls from target_links
    for target in soup.find_all(target_links):
        # Get all the links in target
        for link in target.find_all('a'):
            child_urls.append(urllib.parse.urljoin(base_url, link.get('href')))

    result = []
    for u in child_urls:

        child_url = urllib.parse.urlparse(u)

        # ---------------------------------
        # Check if domain match is required
        # ---------------------------------
        if is_domain_match and child_url.netloc != urllib.parse.urlparse(base_url).netloc:
            continue

        if is_base_path_match and child_url.path.startswith(urllib.parse.urlparse(base_url).path):
            result.append(u)
            continue

        if valid_paths:
            for valid_path in valid_paths:
                if child_url.path.startswith(urllib.parse.urlparse(valid_path).path):
                    result.append(u)
                    break

    return result


# ------------------
# Worker thread logic
# ------------------
def worker(
    q: object,
    base_url: str,
    max_depth: int,
    already_crawled: set,
    base_dir: str,
    stop_flag: threading.Event,
    target_links: Union[List[str], None] = DEFAULT_TARGET_LINKS,
    target_content: Union[List[str], None] = None,
    valid_paths: Union[List[str], None] = None,
    is_domain_match: bool = None,
    is_base_path_match: bool = None,
    is_links: Optional[bool] = False
) -> None:

    while not q.empty() and not stop_flag.is_set():
        try:
            depth, url = q.get(timeout=1)  # Add timeout to allow checking stop_flag
        except queue.Empty:
            continue
        
        logger.debug(f'Retrieved URL: {url} from queue')

        if depth > max_depth or stop_flag.is_set():
            continue

        # Create a more unique filename using the full URL including domain
        parsed_url = urllib.parse.urlparse(url)
        
        # Start with the domain (netloc)
        domain_parts = parsed_url.netloc.split('.')
        path_parts = [part for part in parsed_url.path.split('/') if part]  # Remove empty parts
        
        # Combine domain and path parts
        all_parts = domain_parts + path_parts
        
        if all_parts:
            # Join all parts with underscores and clean for filesystem
            file_name = '_'.join(all_parts)
            # Remove or replace invalid filename characters
            file_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
            # Replace multiple underscores with single ones
            file_name = re.sub(r'_+', '_', file_name)
        else:
            file_name = 'index'
            
        file_path = f'{base_dir.rstrip("/") + "/"}{file_name}.md'

        child_urls = crawl(
            url,
            base_url,
            already_crawled,
            file_path,
            target_links,
            target_content,
            valid_paths,
            is_domain_match,
            is_base_path_match,
            is_links
        )
        child_urls = [normalize_url(u) for u in child_urls]
        logger.debug(f'Processing {len(child_urls)} child_urls into queue (current size: {q.qsize()})')
        for child_url in child_urls:
            if not stop_flag.is_set() and child_url not in already_crawled:
                q.put((depth + 1, child_url))
        logger.debug(f'Done processing child_urls into queue (current size: {q.qsize()})')
        #time.sleep(1)


# -----------------
# Thread management
# -----------------
def md_crawl(
        base_url: str,
        max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
        num_threads: Optional[int] = DEFAULT_NUM_THREADS,
        base_dir: Optional[str] = DEFAULT_BASE_DIR,
        target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
        target_content: Union[str, List[str]] = None,
        valid_paths: Union[str, List[str]] = None,
        is_domain_match: Optional[bool] = None,
        is_base_path_match: Optional[bool] = None,
        is_debug: Optional[bool] = False,
        is_links: Optional[bool] = False
) -> None:
    if is_domain_match is False and is_base_path_match is True:
        raise ValueError('❌ Domain match must be True if base match is set to True')

    is_domain_match = DEFAULT_DOMAIN_MATCH if is_domain_match is None else is_domain_match
    is_base_path_match = DEFAULT_BASE_PATH_MATCH if is_base_path_match is None else is_base_path_match

    if not base_url:
        raise ValueError('❌ Base URL is required')

    if isinstance(target_links, str):
        target_links = target_links.split(',') if ',' in target_links else [target_links]

    if isinstance(target_content, str):
        target_content = target_content.split(',') if ',' in target_content else [target_content]

    if isinstance(valid_paths, str):
        valid_paths = valid_paths.split(',') if ',' in valid_paths else [valid_paths]

    if is_debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug('🐞 Debugging enabled')
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f'🕸️ Crawling {base_url} at ⏬ depth {max_depth} with 🧵 {num_threads} threads')

    # Validate the base URL
    if not is_valid_url(base_url):
        raise ValueError('❌ Invalid base URL')

    # Create base_dir if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    already_crawled = set()
    stop_flag = threading.Event()

    # Create a queue of URLs to crawl with special handling to avoid duplicate URLs
    q = URLQueue()

    # Add the base URL to the queue
    q.put((0, base_url))

    threads = []

    # Create a thread for each URL in the queue
    for i in range(num_threads):
        t = threading.Thread(
            target=worker,
            args=(
                q,
                base_url,
                max_depth,
                already_crawled,
                base_dir,
                stop_flag,
                target_links,
                target_content,
                valid_paths,
                is_domain_match,
                is_base_path_match,
                is_links
            )
        )
        t.daemon = True  # Make thread daemon so it dies when main thread exits
        threads.append(t)
        t.start()
        logger.debug(f'Started thread {i+1} of {num_threads}')

    try:
        # Wait for all threads to finish or until interrupted
        while any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info('🛑 Interrupted by user, stopping crawl...')
        stop_flag.set()
        
        # Give threads a moment to stop gracefully
        for t in threads:
            t.join(timeout=2)

    logger.info('🏁 All threads have finished')
