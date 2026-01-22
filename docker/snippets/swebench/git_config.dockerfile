# SWE-Bench Git Configuration Snippet
# Purpose: Configure git for cleaner output in agent environments
# Source: SkyRL-Agent utils.py:167
#
# These settings prevent git from using a pager (which can cause issues in
# non-interactive environments) and disable binary diff display.

RUN git config --global core.pager "" && \
    git config --global diff.binary false
