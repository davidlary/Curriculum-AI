"""
Terminal UI module for OpenBooks - Beautiful, colorful, and informative progress display.

This module provides enhanced terminal output with colors, progress indicators,
and structured information display for better user experience.
"""

import sys
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output optimized for white backgrounds."""
    
    # Professional colors for white backgrounds (256-color mode)
    RED = '\033[38;5;88m'           # Deep burgundy red
    GREEN = '\033[38;5;22m'         # Forest green  
    YELLOW = '\033[38;5;94m'        # Dark golden brown
    BLUE = '\033[38;5;18m'          # Navy blue
    MAGENTA = '\033[38;5;53m'       # Deep purple
    CYAN = '\033[38;5;23m'          # Teal
    WHITE = '\033[38;5;15m'         # Pure white
    BLACK = '\033[38;5;16m'         # Pure black
    GRAY = '\033[38;5;244m'         # Medium gray
    
    # Professional accent colors
    BRIGHT_RED = '\033[38;5;160m'   # Clear red for errors
    BRIGHT_GREEN = '\033[38;5;34m'  # Vibrant green for success
    BRIGHT_BLUE = '\033[38;5;27m'   # Clear blue for info
    ORANGE = '\033[38;5;166m'       # Orange for warnings
    PURPLE = '\033[38;5;55m'        # Deep purple for headers
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Reset
    RESET = '\033[0m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    
    @staticmethod
    def disable():
        """Disable colors for non-terminal output."""
        for attr in dir(Colors):
            if not attr.startswith('_') and attr != 'disable':
                setattr(Colors, attr, '')


class Symbols:
    """Unicode symbols for better visual representation."""
    
    # Status symbols
    SUCCESS = '✅'
    ERROR = '❌'
    WARNING = '⚠️'
    INFO = 'ℹ️'
    PROCESSING = '⚙️'
    SEARCHING = '🔍'
    DOWNLOADING = '⬇️'
    
    # Progress symbols
    ARROW_RIGHT = '→'
    BULLET = '•'
    CHECKMARK = '✓'
    CROSS = '✗'
    
    # Category symbols
    BOOK = '📚'
    LANGUAGE = '🌍'
    FOLDER = '📁'
    GIT = '🔗'
    STATS = '📊'


@dataclass
class ProgressInfo:
    """Information for progress display."""
    current: int
    total: int
    operation: str
    details: str = ""
    start_time: float = 0.0


class TerminalUI:
    """Enhanced terminal interface for OpenBooks operations."""
    
    def __init__(self, verbose: bool = False, no_color: bool = False):
        """Initialize terminal UI."""
        self.verbose = verbose
        self.start_time = time.time()
        self.last_update = 0
        self._lock = threading.Lock()  # Thread safety for output
        self._current_books = {}  # Track currently processing books
        
        # Disable colors if requested or if not a TTY
        if no_color or not sys.stdout.isatty():
            Colors.disable()
    
    def _safe_print(self, message: str, end: str = '\n', flush: bool = False):
        """Thread-safe printing with lock."""
        with self._lock:
            print(message, end=end, flush=flush)
    
    def _display_width(self, text: str) -> int:
        """Calculate the visual display width of text, accounting for emoji characters."""
        # Simple heuristic: count emoji characters as 2 display columns
        import re
        # Unicode emoji range patterns
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        
        # Count regular characters
        display_width = len(text)
        
        # Add extra width for emojis (each emoji takes 2 columns but len() counts as 1)
        emoji_matches = emoji_pattern.findall(text)
        for match in emoji_matches:
            display_width += len(match)  # Add 1 extra for each emoji character
        
        return display_width

    def print_header(self, title: str, subtitle: str = ""):
        """Print an elegant header with gradient effect."""
        width = 80
        
        # Create elegant border with Unicode box drawing characters
        top_border = f"{Colors.PURPLE}{Colors.BOLD}╔{'═' * (width-2)}╗{Colors.RESET}"
        bottom_border = f"{Colors.PURPLE}{Colors.BOLD}╚{'═' * (width-2)}╝{Colors.RESET}"
        
        # Calculate padding manually using display width
        title_display_width = self._display_width(title)
        title_padding = width - 2 - title_display_width
        left_pad = title_padding // 2
        right_pad = title_padding - left_pad
        centered_title = " " * left_pad + title + " " * right_pad
        
        header_lines = [
            f"\n{top_border}",
            f"{Colors.PURPLE}{Colors.BOLD}║{Colors.BLACK}{Colors.BOLD}{centered_title}{Colors.PURPLE}{Colors.BOLD}║{Colors.RESET}"
        ]
        
        if subtitle:
            subtitle_display_width = self._display_width(subtitle)
            subtitle_padding = width - 2 - subtitle_display_width
            sub_left_pad = subtitle_padding // 2
            sub_right_pad = subtitle_padding - sub_left_pad
            centered_subtitle = " " * sub_left_pad + subtitle + " " * sub_right_pad
            header_lines.append(f"{Colors.PURPLE}{Colors.BOLD}║{Colors.GRAY}{centered_subtitle}{Colors.PURPLE}{Colors.BOLD}║{Colors.RESET}")
        
        header_lines.extend([
            bottom_border,
            ""
        ])
        
        with self._lock:
            for line in header_lines:
                print(line)
    
    def print_phase(self, phase_num: int, phase_name: str, description: str = ""):
        """Print an elegant phase header with visual separator."""
        # Create visual separator with gradient
        separator = f"{Colors.CYAN}{Colors.BOLD}{'─' * 20} ◆ {'─' * 20}{Colors.RESET}"
        
        lines = [
            f"\n{separator}",
            f"{Colors.BLUE}{Colors.BOLD}▶ Phase {phase_num}: {phase_name}{Colors.RESET}"
        ]
        if description:
            lines.append(f"{Colors.GRAY}  {description}{Colors.RESET}")
        lines.append("")
        
        with self._lock:
            for line in lines:
                print(line)
    
    def print_success(self, message: str, details: str = ""):
        """Print an elegant success message."""
        lines = [f"{Colors.BRIGHT_GREEN}{Symbols.SUCCESS} {message}{Colors.RESET}"]
        if details and self.verbose:
            lines.append(f"{Colors.GRAY}   └─ {details}{Colors.RESET}")
        
        with self._lock:
            for line in lines:
                print(line)
    
    def print_error(self, message: str, details: str = ""):
        """Print an elegant error message."""
        lines = [f"{Colors.BRIGHT_RED}{Symbols.ERROR} {message}{Colors.RESET}"]
        if details:
            lines.append(f"{Colors.GRAY}   └─ {details}{Colors.RESET}")
        
        with self._lock:
            for line in lines:
                print(line)
    
    def print_warning(self, message: str, details: str = ""):
        """Print an elegant warning message."""
        lines = [f"{Colors.ORANGE}{Symbols.WARNING} {message}{Colors.RESET}"]
        if details and self.verbose:
            lines.append(f"{Colors.GRAY}   └─ {details}{Colors.RESET}")
        
        with self._lock:
            for line in lines:
                print(line)
    
    def print_info(self, message: str, details: str = ""):
        """Print an elegant info message."""
        if self.verbose:
            lines = [f"{Colors.BRIGHT_BLUE}{Symbols.INFO} {message}{Colors.RESET}"]
            if details:
                lines.append(f"{Colors.GRAY}   └─ {details}{Colors.RESET}")
            
            with self._lock:
                for line in lines:
                    print(line)
    
    def print_log_notification(self, level: str, category: str = "", brief_msg: str = ""):
        """Print notification that something was written to log files."""
        level_colors = {
            'error': Colors.RED,
            'warning': Colors.YELLOW,
            'critical': Colors.BRIGHT_RED,
            'info': Colors.CYAN,
            'debug': Colors.GRAY
        }
        
        level_symbols = {
            'error': '📝',
            'warning': '📝', 
            'critical': '📝',
            'info': '📝',
            'debug': '📝'
        }
        
        color = level_colors.get(level.lower(), Colors.GRAY)
        symbol = level_symbols.get(level.lower(), '📝')
        
        if category:
            msg = f"{color}{symbol} {level.upper()}{Colors.RESET} logged to file"
            if brief_msg:
                msg += f"{Colors.GRAY} - {brief_msg}{Colors.RESET}"
            msg += f"{Colors.GRAY} (check openbooks.log){Colors.RESET}"
        else:
            msg = f"{color}{symbol} {level.upper()}{Colors.RESET}{Colors.GRAY} logged to openbooks.log{Colors.RESET}"
        
        with self._lock:
            print(msg)
    
    def print_book_processing(self, book_name: str, source: str, operation: str):
        """Print information about a book being processed."""
        with self._lock:
            self._current_books[book_name] = {
                'source': source,
                'operation': operation,
                'start_time': time.time()
            }
            
            # Format the source for display
            source_display = source.replace('https://', '').replace('http://', '')
            if len(source_display) > 50:
                source_display = source_display[:47] + "..."
            
            message = f"{Symbols.PROCESSING} {operation}: {Colors.BOLD}{book_name}{Colors.RESET}"
            details = f"{Colors.GRAY}Source: {source_display}{Colors.RESET}"
            
            print(f"{message} - {details}")
    
    def print_book_completed(self, book_name: str, status: str, details: str = ""):
        """Print completion status for a book."""
        with self._lock:
            if book_name in self._current_books:
                book_info = self._current_books.pop(book_name)
                duration = time.time() - book_info['start_time']
                
                status_colors = {
                    'success': Colors.GREEN,
                    'exists': Colors.YELLOW,
                    'error': Colors.RED,
                    'skipped': Colors.DIM
                }
                
                status_symbols = {
                    'success': Symbols.SUCCESS,
                    'exists': Symbols.INFO,
                    'error': Symbols.ERROR,
                    'skipped': Symbols.BULLET
                }
                
                color = status_colors.get(status, Colors.WHITE)
                symbol = status_symbols.get(status, Symbols.BULLET)
                
                message = f"{color}{symbol} {book_name}{Colors.RESET}"
                if details:
                    message += f" {Colors.DIM}({details}, {duration:.1f}s){Colors.RESET}"
                else:
                    message += f" {Colors.DIM}({duration:.1f}s){Colors.RESET}"
                
                print(message)
    
    def print_progress(self, current: int, total: int, operation: str, details: str = ""):
        """Print an elegant progress bar with gradient colors."""
        # Throttle updates to avoid spam
        now = time.time()
        if now - self.last_update < 0.1 and current < total:
            return
        self.last_update = now
        
        if total == 0:
            percent = 100
        else:
            percent = (current / total) * 100
        
        # Elegant progress bar with Unicode characters
        bar_width = 25
        filled = int(bar_width * current / total) if total > 0 else bar_width
        
        # Create gradient progress bar
        bar_chars = []
        for i in range(bar_width):
            if i < filled:
                if percent < 25:
                    bar_chars.append(f"{Colors.BRIGHT_RED}█{Colors.RESET}")
                elif percent < 50:
                    bar_chars.append(f"{Colors.ORANGE}█{Colors.RESET}")
                elif percent < 75:
                    bar_chars.append(f"{Colors.YELLOW}█{Colors.RESET}")
                else:
                    bar_chars.append(f"{Colors.BRIGHT_GREEN}█{Colors.RESET}")
            else:
                bar_chars.append(f"{Colors.GRAY}░{Colors.RESET}")
        
        bar = ''.join(bar_chars)
        
        # Format elegant message
        status_msg = f"{Colors.BLUE}{Colors.BOLD}{operation}{Colors.RESET}"
        progress_msg = f"{Colors.BLACK}[{bar}] {Colors.BOLD}{percent:5.1f}%{Colors.RESET} {Colors.GRAY}({current}/{total}){Colors.RESET}"
        
        with self._lock:
            # Get terminal width for proper clearing
            try:
                import os
                terminal_width = os.get_terminal_size().columns
            except:
                terminal_width = 80  # fallback
            
            # Clear the entire line properly
            print('\r' + ' ' * terminal_width + '\r', end='', flush=True)
            
            if details:
                # Truncate details if too long to prevent overrun
                max_details_length = 50
                if len(details) > max_details_length:
                    details = details[:max_details_length-3] + "..."
                print(f"\r{status_msg} {progress_msg} {Colors.GRAY}│ {details}{Colors.RESET}", end='', flush=True)
            else:
                print(f"\r{status_msg} {progress_msg}", end='', flush=True)
            
            if current >= total:
                print()  # New line when complete
    
    def print_statistics(self, stats: Dict[str, Any]):
        """Print beautifully formatted statistics with visual hierarchy."""
        BOX_WIDTH = 50
        CONTENT_WIDTH = BOX_WIDTH - 2  # Account for left and right borders (│ and │)
        
        # Create elegant stats header
        print(f"\n{Colors.CYAN}{Colors.BOLD}┌{'─' * (BOX_WIDTH-2)}┐{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}│{f'{Symbols.STATS} Collection Statistics'.center(BOX_WIDTH-3)}│{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}├{'─' * (BOX_WIDTH-2)}┤{Colors.RESET}")
        
        for key, value in stats.items():
            if isinstance(value, dict):
                # Header for subcategory
                header_text = f" {key}:"
                # Strip any ANSI codes and ensure exact fit
                clean_header = self._strip_ansi_codes(header_text)
                if len(clean_header) > CONTENT_WIDTH:
                    clean_header = clean_header[:CONTENT_WIDTH-3] + "..."
                # Pad to EXACTLY content width using clean text length
                padded_header = clean_header + " " * (CONTENT_WIDTH - len(clean_header))
                print(f"{Colors.CYAN}│{Colors.BLACK}{Colors.BOLD}{padded_header}{Colors.RESET}{Colors.CYAN}│{Colors.RESET}")
                
                for sub_key, sub_value in value.items():
                    formatted_value = f"{sub_value:.1f}" if isinstance(sub_value, float) else str(sub_value)
                    sub_text = f"   └─ {sub_key}: {formatted_value}"
                    # Strip any ANSI codes and ensure exact fit
                    clean_sub = self._strip_ansi_codes(sub_text)
                    if len(clean_sub) > CONTENT_WIDTH:
                        clean_sub = clean_sub[:CONTENT_WIDTH-3] + "..."
                    # Pad to EXACTLY content width using clean text length
                    padded_sub = clean_sub + " " * (CONTENT_WIDTH - len(clean_sub))
                    print(f"{Colors.CYAN}│{Colors.GRAY}{padded_sub}{Colors.RESET}{Colors.CYAN}│{Colors.RESET}")
            else:
                # Format numbers elegantly
                if isinstance(value, float):
                    formatted_value = f"{value:.1f}"
                elif isinstance(value, int) and value > 1000:
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)
                
                # Calculate exact spacing for alignment
                main_text = f" {key}: {formatted_value}"
                # Strip any ANSI codes and ensure exact fit
                clean_main = self._strip_ansi_codes(main_text)
                if len(clean_main) > CONTENT_WIDTH:
                    clean_main = clean_main[:CONTENT_WIDTH-3] + "..."
                # Pad to EXACTLY content width using clean text length
                padded_main = clean_main + " " * (CONTENT_WIDTH - len(clean_main))
                print(f"{Colors.CYAN}│{Colors.BLACK}{padded_main}{Colors.RESET}{Colors.CYAN}│{Colors.RESET}")
        
        print(f"{Colors.CYAN}{Colors.BOLD}└{'─' * (BOX_WIDTH-2)}┘{Colors.RESET}\n")
    
    def print_language_summary(self, languages: Dict[str, int]):
        """Print elegant language distribution with visual bars."""
        if not languages:
            return
        
        BOX_WIDTH = 50
        CONTENT_WIDTH = BOX_WIDTH - 2  # Account for left and right borders (│ and │)
        
        # Create elegant language header    
        print(f"\n{Colors.PURPLE}{Colors.BOLD}┌{'─' * (BOX_WIDTH-2)}┐{Colors.RESET}")
        print(f"{Colors.PURPLE}{Colors.BOLD}│{f'{Symbols.LANGUAGE} Language Distribution'.center(BOX_WIDTH-3)}│{Colors.RESET}")
        print(f"{Colors.PURPLE}{Colors.BOLD}├{'─' * (BOX_WIDTH-2)}┤{Colors.RESET}")
        
        total = sum(languages.values())
        
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            percent = (count / total) * 100 if total > 0 else 0
            
            # Create a simpler, more predictable format
            lang_display = f" {lang.capitalize()[:11]:11}"  # Language name, max 11 chars
            count_display = f"{count:3d}"  # Count, 3 chars
            percent_display = f"{percent:5.1f}%"  # Percentage, 6 chars
            
            # Simple bar without color codes for easier alignment (15 chars)
            bar_length = max(1, min(15, int(percent / 100 * 15)))  # Scale to 15 chars max
            bar_display = "█" * bar_length + "░" * (15 - bar_length)
            
            # Construct the line: " Language   123 ███████████░░░░ 100.0% "
            line_content = f"{lang_display} {count_display} {bar_display} {percent_display}"
            
            # Strip any ANSI codes and ensure exact fit
            clean_content = self._strip_ansi_codes(line_content)
            if len(clean_content) > CONTENT_WIDTH:
                clean_content = clean_content[:CONTENT_WIDTH]
            # Pad to EXACTLY content width using clean text length
            padded_content = clean_content + " " * (CONTENT_WIDTH - len(clean_content))
            
            print(f"{Colors.PURPLE}│{Colors.BLACK}{padded_content}{Colors.RESET}{Colors.PURPLE}│{Colors.RESET}")
        
        print(f"{Colors.PURPLE}{Colors.BOLD}└{'─' * (BOX_WIDTH-2)}┘{Colors.RESET}\n")
    
    def print_repository_status(self, repo_name: str, status: str, details: str = ""):
        """Print repository operation status."""
        status_colors = {
            'cloning': Colors.BLUE,
            'success': Colors.GREEN, 
            'exists': Colors.YELLOW,
            'error': Colors.RED,
            'skipped': Colors.DIM
        }
        
        status_symbols = {
            'cloning': Symbols.DOWNLOADING,
            'success': Symbols.SUCCESS,
            'exists': Symbols.INFO,
            'error': Symbols.ERROR,
            'skipped': Symbols.BULLET
        }
        
        color = status_colors.get(status, Colors.WHITE)
        symbol = status_symbols.get(status, Symbols.BULLET)
        
        if not self.verbose and status == 'exists':
            return  # Don't show "already exists" in non-verbose mode
        
        print(f"{color}{symbol} {repo_name}{Colors.RESET}", end='')
        if details:
            print(f" {Colors.DIM}({details}){Colors.RESET}")
        else:
            print()
    
    def print_elapsed_time(self):
        """Print elapsed time since initialization."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        if minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        
        print(f"\n{Colors.WHITE}{Colors.DIM}⏱️  Total time: {time_str}{Colors.RESET}")
    
    def print_operation_summary(self, operation: str, success_count: int, 
                              total_count: int, errors: List[str] = None):
        """Print operation summary."""
        print(f"\n{Colors.BOLD}{operation} Summary:{Colors.RESET}")
        
        if success_count == total_count:
            print(f"{Colors.GREEN}{Symbols.SUCCESS} All {total_count} operations completed successfully{Colors.RESET}")
        else:
            failed_count = total_count - success_count
            print(f"{Colors.GREEN}{Symbols.SUCCESS} {success_count} successful{Colors.RESET}")
            if failed_count > 0:
                print(f"{Colors.RED}{Symbols.ERROR} {failed_count} failed{Colors.RESET}")
        
        if errors and self.verbose:
            print(f"\n{Colors.RED}Errors encountered:{Colors.RESET}")
            for error in errors[:5]:  # Show max 5 errors
                print(f"  {Colors.RED}{Symbols.BULLET} {error}{Colors.RESET}")
            if len(errors) > 5:
                print(f"  {Colors.DIM}... and {len(errors) - 5} more errors{Colors.RESET}")
    
    def print_dry_run_notice(self):
        """Print dry run notice."""
        print(f"\n{Colors.YELLOW}{Colors.BOLD}{Symbols.INFO} DRY RUN MODE - No changes will be made{Colors.RESET}\n")
    
    def print_config_info(self, config: Dict[str, Any]):
        """Print configuration information."""
        if not self.verbose:
            return
            
        print(f"{Colors.CYAN}{Colors.DIM}Configuration:{Colors.RESET}")
        for key, value in config.items():
            if isinstance(value, bool):
                value_str = f"{Colors.GREEN}✓{Colors.RESET}" if value else f"{Colors.RED}✗{Colors.RESET}"
            else:
                value_str = f"{Colors.WHITE}{value}{Colors.RESET}"
            print(f"  {Colors.DIM}{key}: {value_str}")
        print()
    
    def clear_line(self):
        """Clear the current line."""
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except:
            terminal_width = 80  # fallback
        print('\r' + ' ' * terminal_width + '\r', end='', flush=True)
    
    def _strip_ansi_codes(self, text: str) -> str:
        """Strip ANSI color codes from text to get actual display length."""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def print_separator(self, char: str = '─', length: int = 50):
        """Print a separator line."""
        print(f"{Colors.GRAY}{char * length}{Colors.RESET}")
    
    def print_error_warning_summary(self, error_count: int, warning_count: int):
        """Print a prominent summary of errors and warnings if any occurred."""
        if error_count > 0 or warning_count > 0:
            print(f"\n{Colors.BRIGHT_RED}{Colors.BOLD}{'!' * 60}{Colors.RESET}")
            print(f"{Colors.BRIGHT_RED}{Colors.BOLD}   ATTENTION: ERRORS/WARNINGS OCCURRED DURING PROCESSING   {Colors.RESET}")
            print(f"{Colors.BRIGHT_RED}{Colors.BOLD}{'!' * 60}{Colors.RESET}")
            
            if error_count > 0:
                print(f"{Colors.RED}{Colors.BOLD}❌ {error_count} ERROR(S) logged to openbooks.log{Colors.RESET}")
            
            if warning_count > 0:
                print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  {warning_count} WARNING(S) logged to openbooks.log{Colors.RESET}")
            
            print(f"{Colors.GRAY}📝 Please check the log file for detailed information:{Colors.RESET}")
            print(f"{Colors.GRAY}   tail -f openbooks.log{Colors.RESET}")
            print(f"{Colors.BRIGHT_RED}{Colors.BOLD}{'!' * 60}{Colors.RESET}\n")


def create_ui(verbose: bool = False, no_color: bool = False) -> TerminalUI:
    """Factory function to create a TerminalUI instance."""
    return TerminalUI(verbose=verbose, no_color=no_color)