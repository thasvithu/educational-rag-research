# ================================================================
# Script  : explore_pdfs.py
# Step    : 01 — Load & Explore PDFs
#
# Purpose : Load each PDF from a team member's data folder using
#           LangChain's PyPDFLoader, then run a detailed character
#           analysis to understand the raw text quality BEFORE
#           any cleaning is applied.
#
#           This is an educational step — team members can see
#           exactly what LangChain extracts from a PDF and which
#           unwanted characters, repeated lines, or encoding
#           artifacts are present in the raw text.
#
# Usage   :
#   1. Set MEMBER_NAME below to your name
#   2. Run:  python explore_pdfs.py
#
# Output  :
#   - Raw text preview for each PDF
#   - Character frequency analysis (unusual, non-ASCII, etc.)
#   - Repeated line detection (potential headers/footers)
#   - Final summary table comparing all PDFs
#   - A plain-text report file saved to:
#       data/outputs/explore_pdf_output_{member_name}.txt
# ================================================================

import string
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule


# ================================================================
# ▶  CONFIGURATION — Each team member changes only this line
# ================================================================

MEMBER_NAME = "vithusan"   # Options: "vithusan" | "aysha" | "haleema"

# How many characters of raw text to preview for each PDF.
# Increase this number if you want to see more of the raw output.
TEXT_PREVIEW_LENGTH = 600

# ================================================================
# Path Setup — Do NOT change anything below this line
# ================================================================

# Resolve the project root (one level above this script's folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to this member's PDF folder: project_root/data/{member_name}/
DATA_DIR = PROJECT_ROOT / "data" / MEMBER_NAME

# Path where output report files are saved: project_root/data/outputs/
# Each team member's report is saved as a separate file so they
# can open and compare results without running the script again.
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"


# ================================================================
# Helper Function 1: Find PDF Files
# ================================================================

def find_pdf_files(folder: Path) -> list[Path]:
    """
    Scan the given folder and return all valid PDF files.

    This function handles two cases:
      1. Files with the ".pdf" extension (normal case).
      2. Files WITHOUT the ".pdf" extension that are still valid
         PDFs — detected by checking the first 4 bytes of the file
         for the PDF magic header "%PDF".

    Args:
        folder (Path): The folder to search for PDF files.

    Returns:
        list[Path]: Sorted list of Path objects for each PDF found.
    """
    pdf_files = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue  # Skip subdirectories

        if file_path.suffix.lower() == ".pdf":
            # Standard case: file has a .pdf extension
            pdf_files.append(file_path)
        else:
            # Edge case: file has no extension — check the PDF magic bytes
            # All valid PDF files start with the 4-byte signature: %PDF
            try:
                with open(file_path, "rb") as f:
                    header = f.read(4)
                if header == b"%PDF":
                    pdf_files.append(file_path)
            except OSError:
                pass  # Skip files that cannot be read

    return pdf_files


# ================================================================
# Helper Function 2: Analyze Characters
# ================================================================

def analyze_characters(full_text: str) -> dict:
    """
    Perform a detailed character-level analysis of extracted PDF text.

    This function identifies four categories of problematic content
    that are common in university handbook PDFs:

      1. Unusual characters  — symbols not part of standard English text
      2. Non-ASCII characters — characters outside the ASCII range (0-127),
                                such as Sinhala/Tamil script or math symbols
      3. Single-letter tokens — isolated single letters that are often
                                formatting artifacts (e.g., "s", "x", "n")
      4. Repeated lines       — lines that appear many times, usually
                                page headers or footers

    Args:
        full_text (str): The complete concatenated text from all PDF pages.

    Returns:
        dict: A dictionary containing all analysis results.
    """

    # Count every unique character in the entire text
    char_counts = Counter(full_text)

    # ---- 1. Unusual Characters ----------------------------------------
    # Standard characters = English letters + digits + spaces + punctuation
    # Anything outside this set is considered "unusual" for English text.
    standard_chars = set(
        string.ascii_letters +   # a-z, A-Z
        string.digits +          # 0-9
        string.whitespace +      # spaces, tabs, newlines
        string.punctuation       # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    )
    unusual_chars = {
        char: count
        for char, count in char_counts.items()
        if char not in standard_chars
    }

    # ---- 2. Non-ASCII Characters ---------------------------------------
    # Characters with Unicode code point > 127 are outside ASCII range.
    # In Sri Lankan educational documents, these are commonly:
    #   - Sinhala script characters (U+0D80 to U+0DFF)
    #   - Tamil script characters  (U+0B80 to U+0BFF)
    #   - Mathematical symbols     (e.g., ∑ α β π ∫)
    #   - Encoding artifacts       (e.g., â€™ instead of apostrophe)
    non_ascii_chars = {
        char: count
        for char, count in char_counts.items()
        if ord(char) > 127
    }

    # ---- 3. Single-Letter Tokens ---------------------------------------
    # Split text by whitespace to get individual tokens (words).
    # Single alphabetic tokens like "a" and "I" are valid English words,
    # but others (e.g., "s", "x", "z") are usually noise from:
    #   - Bullet point formatting being extracted as a lone letter
    #   - Column layout bleeding between columns
    #   - Table cell content being isolated
    tokens = full_text.split()
    single_letter_tokens = [token for token in tokens
                            if len(token) == 1 and token.isalpha()]

    # ---- 4. Repeated Lines (Headers / Footers) -------------------------
    # Split text into lines and count how many times each line appears.
    # Lines appearing more than 3 times AND longer than 3 characters
    # are almost certainly page headers or footers — noise to be removed.
    lines = full_text.split("\n")
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    line_counts = Counter(non_empty_lines)
    repeated_lines = {
        line: count
        for line, count in line_counts.items()
        if count > 3 and len(line) > 3
    }

    # Return all results in a structured dictionary
    return {
        "total_characters"  : len(full_text),
        "total_words"       : len(tokens),
        "total_lines"       : len(non_empty_lines),
        "unusual_chars"     : unusual_chars,
        "non_ascii_chars"   : non_ascii_chars,
        "single_letter_tokens" : single_letter_tokens,
        "repeated_lines"    : repeated_lines,
        # Top 20 unusual characters sorted by frequency (most common first)
        "top_unusual"       : sorted(
                                unusual_chars.items(),
                                key=lambda item: item[1],
                                reverse=True
                              )[:20],
    }


# ================================================================
# Helper Function 3: Load and Analyze One PDF
# ================================================================

def load_and_analyze_pdf(pdf_path: Path, console: Console) -> dict | None:
    """
    Load a single PDF using LangChain's PyPDFLoader and display
    a full analysis of its extracted text content.

    How LangChain loads PDFs:
      - PyPDFLoader reads the PDF file page by page.
      - Each page becomes a LangChain Document object with:
          * page_content : the extracted text string
          * metadata     : page number, source file path, etc.
      - We then join all page texts together for analysis.

    Args:
        pdf_path (Path): Full path to the PDF file to load.
        console  (Console): Rich console object for formatted output.

    Returns:
        dict: Summary statistics for this PDF (used in final report).
        None: If the PDF could not be loaded.
    """

    # Section divider for this PDF
    console.rule(f"[bold cyan]📄  {pdf_path.name}[/bold cyan]")

    try:
        # ----------------------------------------------------------
        # Step A: Load the PDF with LangChain's PyPDFLoader
        # ----------------------------------------------------------
        # PyPDFLoader uses the pypdf library underneath to open the
        # file and extract text from each page individually.
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # Returns a list of Document objects

        # Check if any text was actually extracted
        if not pages:
            console.print(
                "[bold red]⚠ WARNING:[/bold red] No text was extracted from this PDF.\n"
                "   This usually means the PDF is image-based (scanned) and requires OCR.\n"
                "   Please remove this file from the dataset.\n"
            )
            return None

        console.print(
            f"[green]✓ Loaded successfully[/green] — "
            f"[bold]{len(pages)} page(s)[/bold] extracted by LangChain"
        )

        # ----------------------------------------------------------
        # Step B: Combine all pages into a single text block
        # ----------------------------------------------------------
        # We join the content of all pages with a newline separator.
        # This gives us the complete raw document text for analysis.
        full_text = "\n".join(page.page_content for page in pages)

        # ----------------------------------------------------------
        # Step C: Show the Raw Text Preview
        # ----------------------------------------------------------
        # This is the most important part for team members to see —
        # exactly what LangChain extracts from the PDF in raw form.
        # Notice how headers, footers, and special characters appear.
        console.print(
            f"\n[bold yellow]📝 Raw Text Preview "
            f"(first {TEXT_PREVIEW_LENGTH} characters):[/bold yellow]"
        )
        # Replace newline characters with a visible "↵" symbol so
        # team members can clearly see where line breaks occur.
        preview_text = full_text[:TEXT_PREVIEW_LENGTH].replace("\n", " ↵\n")
        console.print(Panel(preview_text, border_style="yellow", expand=False))

        # ----------------------------------------------------------
        # Step D: Run Character Analysis
        # ----------------------------------------------------------
        console.print("\n[bold magenta]🔬 Character Analysis Results:[/bold magenta]")
        analysis = analyze_characters(full_text)

        # ---- Basic Statistics Table ----
        stats_table = Table(show_header=True, header_style="bold blue", box=None)
        stats_table.add_column("Metric",  style="cyan",  min_width=40)
        stats_table.add_column("Value",   style="white", justify="right")

        stats_table.add_row("Total Characters Extracted",
                            f"{analysis['total_characters']:,}")
        stats_table.add_row("Total Words (tokens)",
                            f"{analysis['total_words']:,}")
        stats_table.add_row("Total Non-Empty Lines",
                            f"{analysis['total_lines']:,}")
        stats_table.add_row("─" * 40, "─" * 10)
        stats_table.add_row("Unique Unusual Character Types",
                            f"[yellow]{len(analysis['unusual_chars'])}[/yellow]")
        stats_table.add_row("Unique Non-ASCII Character Types",
                            f"[red]{len(analysis['non_ascii_chars'])}[/red]")
        stats_table.add_row("Single-Letter Token Count",
                            f"[yellow]{len(analysis['single_letter_tokens'])}[/yellow]")
        stats_table.add_row("Repeated Lines (potential headers/footers)",
                            f"[red]{len(analysis['repeated_lines'])}[/red]")

        console.print(stats_table)

        # ---- Top Unusual Characters Table ----
        if analysis["top_unusual"]:
            console.print(
                "\n[bold red]⚠ Top Unusual Characters Found "
                "(review these for cleaning):[/bold red]"
            )
            unusual_table = Table(show_header=True, header_style="bold red",
                                  show_lines=True)
            unusual_table.add_column("Character",   justify="center", style="white",  width=12)
            unusual_table.add_column("Unicode",     justify="center", style="dim",    width=12)
            unusual_table.add_column("Frequency",   justify="right",  style="yellow", width=12)
            unusual_table.add_column("Type",        style="cyan",                     width=20)

            for char, count in analysis["top_unusual"]:
                code_point = f"U+{ord(char):04X}"

                # Classify the character type to help team understand what it is
                if ord(char) > 127:
                    char_type = "Non-ASCII / Foreign Script"
                elif ord(char) < 32 or ord(char) == 127:
                    char_type = "Control Character"
                else:
                    char_type = "Special Symbol"

                # Some control characters do not print well — show repr instead
                display_char = repr(char) if (ord(char) < 32 or ord(char) == 127) else char

                unusual_table.add_row(display_char, code_point, f"{count:,}", char_type)

            console.print(unusual_table)
        else:
            console.print("[green]✓ No unusual characters found.[/green]")

        # ---- Sample of Single-Letter Tokens ----
        if analysis["single_letter_tokens"]:
            # Show a sample of up to 30 single-letter tokens
            sample = analysis["single_letter_tokens"][:30]
            console.print(
                f"\n[bold yellow]Single-letter tokens sample "
                f"(showing first 30 of {len(analysis['single_letter_tokens'])}):[/bold yellow]"
            )
            console.print(f"  {sample}")

        # ---- Repeated Lines (Headers / Footers) ----
        if analysis["repeated_lines"]:
            console.print(
                "\n[bold red]🔁 Repeated Lines Detected "
                "(likely headers or footers — candidates for removal):[/bold red]"
            )
            repeated_table = Table(show_header=True, header_style="bold red",
                                   show_lines=True)
            repeated_table.add_column("Repeated Line Content", style="white",  min_width=50)
            repeated_table.add_column("Appears N Times",       style="yellow", justify="right", width=16)

            # Show top 10 most-repeated lines
            top_repeated = sorted(
                analysis["repeated_lines"].items(),
                key=lambda item: item[1],
                reverse=True
            )[:10]

            for line, count in top_repeated:
                # Truncate very long lines so the table stays readable
                display_line = (line[:70] + " ...") if len(line) > 70 else line
                repeated_table.add_row(display_line, str(count))

            console.print(repeated_table)
        else:
            console.print("[green]✓ No repeated lines (headers/footers) detected.[/green]")

        console.print()  # Add spacing before the next PDF

        # Return a summary dictionary for use in the final report
        return {
            "pdf_name"             : pdf_path.name,
            "pages"                : len(pages),
            "total_characters"     : analysis["total_characters"],
            "total_words"          : analysis["total_words"],
            "unusual_char_types"   : len(analysis["unusual_chars"]),
            "non_ascii_types"      : len(analysis["non_ascii_chars"]),
            "single_letter_tokens" : len(analysis["single_letter_tokens"]),
            "repeated_lines"       : len(analysis["repeated_lines"]),
        }

    except Exception as error:
        console.print(f"[bold red]✗ Failed to load PDF:[/bold red] {error}\n")
        return None


# ================================================================
# Helper Function 4: Print Final Summary Report
# ================================================================

def save_output_to_file(console: Console) -> None:
    """
    Save the full terminal report to a plain-text file.

    Rich's record mode captures everything printed to the console.
    This function exports that captured text (without color codes)
    to a .txt file in the data/outputs/ folder.

    The output file is named:  explore_pdf_output_{member_name}.txt
    This makes it easy for each team member to keep their own report
    and share it with others via GitHub.

    Args:
        console (Console): The Rich console that recorded all output.
    """
    # Create the outputs folder if it doesn't already exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the output file path with the member's name in the filename
    output_file = OUTPUT_DIR / f"explore_pdf_output_{MEMBER_NAME}.txt"

    # Export all recorded console output as plain text (no color codes)
    # Rich automatically strips ANSI color codes for clean text files.
    console.save_text(str(output_file))

    # Confirm the file was saved (this line prints AFTER recording stops,
    # so it appears in terminal only — not in the saved file)
    print(f"\n✓ Report saved to: {output_file}")


def print_summary_report(results: list[dict], console: Console) -> None:
    """
    Print a final comparison table summarizing all PDFs analyzed.

    This report gives a quick overview of the entire dataset and
    helps identify which PDFs contain the most noise, so the team
    knows where data cleaning is most needed.

    Args:
        results (list[dict]): List of summary dictionaries from each PDF.
        console (Console):    Rich console for formatted output.
    """
    console.rule("[bold green]📊  FINAL SUMMARY REPORT[/bold green]")

    summary_table = Table(
        show_header=True,
        header_style="bold green",
        show_lines=True,
        title=f"PDF Analysis Summary — Member: {MEMBER_NAME}"
    )

    # Define table columns
    summary_table.add_column("PDF File",              style="cyan",   max_width=32)
    summary_table.add_column("Pages",                 justify="right")
    summary_table.add_column("Total Chars",           justify="right")
    summary_table.add_column("Total Words",           justify="right")
    summary_table.add_column("Unusual Char\nTypes",   justify="right", style="yellow")
    summary_table.add_column("Non-ASCII\nTypes",      justify="right", style="red")
    summary_table.add_column("Single-Letter\nTokens", justify="right", style="yellow")
    summary_table.add_column("Repeated\nLines",       justify="right", style="red")

    for result in results:
        # Truncate long file names so the table stays aligned
        name = result["pdf_name"]
        display_name = (name[:30] + "..") if len(name) > 32 else name

        summary_table.add_row(
            display_name,
            str(result["pages"]),
            f"{result['total_characters']:,}",
            f"{result['total_words']:,}",
            str(result["unusual_char_types"]),
            str(result["non_ascii_types"]),
            str(result["single_letter_tokens"]),
            str(result["repeated_lines"]),
        )

    console.print(summary_table)

    # Final notes for the team
    console.print(f"\n[bold]Total PDFs successfully analyzed:[/bold] {len(results)}")
    console.print(
        "\n[bold yellow]📌 What to do next:[/bold yellow]\n"
        "  1. Review the [yellow]'Unusual Char Types'[/yellow] and "
        "[red]'Non-ASCII Types'[/red] columns — high numbers mean more cleaning needed.\n"
        "  2. Note the repeated lines — these are headers/footers to be removed.\n"
        "  3. Once all team members run this script and compare notes,\n"
        "     we will write the [bold]02_clean_data[/bold] script with specific cleaning rules.\n"
    )


# ================================================================
# Main Entry Point
# ================================================================

def main():
    """
    Main function — orchestrates the PDF loading and analysis pipeline.
    Runs automatically when this script is executed.
    """
    # record=True tells Rich to capture everything printed to this console.
    # This allows us to later export the full output to a text file.
    console = Console(record=True)

    # ---- Print Header Banner ----
    console.print(
        Panel.fit(
            f"[bold cyan]Educational RAG Research — PDF Exploration Tool[/bold cyan]\n\n"
            f"  [white]Team Member :[/white] [bold yellow]{MEMBER_NAME}[/bold yellow]\n"
            f"  [white]Data Folder :[/white] {DATA_DIR}\n\n"
            f"  [dim]This script loads each PDF and shows the raw extracted text\n"
            f"  so you can understand the data BEFORE cleaning it.[/dim]",
            border_style="cyan",
            title="[bold]Step 01 — Load & Explore PDFs[/bold]",
            padding=(1, 4),
        )
    )

    # ---- Validate the Data Folder ----
    if not DATA_DIR.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Data folder not found at:\n  {DATA_DIR}\n\n"
            "  Please check that:\n"
            "  1. MEMBER_NAME is spelled correctly (lowercase)\n"
            "  2. Your PDF files are placed inside the correct folder\n"
        )
        return

    # ---- Find All PDF Files ----
    pdf_files = find_pdf_files(DATA_DIR)

    if not pdf_files:
        console.print(
            f"\n[bold red]Error:[/bold red] No PDF files found in:\n  {DATA_DIR}\n\n"
            "  Please add your collected PDF files to this folder and try again.\n"
        )
        return

    console.print(
        f"\n[green]✓ Found [bold]{len(pdf_files)} PDF file(s)[/bold] "
        f"in the data folder.[/green]\n"
    )

    # ---- Process Each PDF One by One ----
    results = []

    for pdf_path in pdf_files:
        result = load_and_analyze_pdf(pdf_path, console)
        if result is not None:
            results.append(result)

    # ---- Print the Final Summary Report ----
    if results:
        print_summary_report(results, console)
    else:
        console.print(
            "[bold red]No PDFs were successfully loaded.[/bold red]\n"
            "Please check that your PDF files are valid and not image-based (scanned).\n"
        )

    # ---- Save the Full Report to a Text File ----
    # This is called AFTER all output has been printed so the file
    # contains the complete report — every PDF's analysis + summary.
    save_output_to_file(console)


# ================================================================
# Script Entry Point Guard
# ================================================================
# This ensures main() is only called when this script is run
# directly (e.g., python explore_pdfs.py), not when it is
# imported as a module by another script.

if __name__ == "__main__":
    main()
