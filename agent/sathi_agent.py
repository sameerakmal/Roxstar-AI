import os
import sys

# Ensure AGENT_NAME is set to roxstar-ai-sathi before importing agent module
os.environ["AGENT_NAME"] = "roxstar-ai-sathi"

from agent import main

if __name__ == "__main__":
    main()
