#!/usr/bin/env python3
"""
Company Knowledge Base Assistant - Main Entry Point
"""

import logging
import sys

from assistant import CompanyKBAssistant
from config import RAG_LOG_LEVEL


def main():
    """Main entry point for the assistant."""
    logging.basicConfig(
        level=getattr(logging, RAG_LOG_LEVEL.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "build-index":
        # Build index mode
        from rag.build_index import build_index
        build_index()
        return
    
    # Interactive Q&A mode
    assistant = CompanyKBAssistant()
    
    print("=" * 60)
    print("🤖 Company Knowledge Base Assistant")
    print("=" * 60)
    print("\nAsk questions about company policies, procedures, and documentation.")
    print("Type 'exit' or 'quit' to stop\n")
    
    try:
        while True:
            query = input("❓ Question: ").strip()
            
            if not query:
                continue
                
            if query.lower() in {"exit", "quit", "q"}:
                print("\n👋 Goodbye!")
                break
            
            print("\n" + "─" * 60)
            print("🤖 Answer:\n")
            
            try:
                result = assistant.query(query, verbose=True)
                print(result["answer"])
                
                if result["sources"]:
                    print("\n📚 Sources:")
                    for src in result["sources"]:
                        print(f"  • {src}")
                
                if result["mcp_used"]:
                    print(f"\n🔧 Used MCP tool: {result['mcp_tool']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            print("─" * 60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
