## README: research_engine/config/

### Workflow Configuration Management

This directory is designated for workflow configuration files used by the research engine. However, it is important to understand the current configuration management strategy:

1. **Authoritative Configuration Location**:
   - The actual workflow configurations are hosted on the **n8n server** at `n8n.bitwiki.org`
   - All active workflows execute from server-stored configurations
   - The server contains the six verified webhook endpoints for:
     - Llama-3.2-3b
     - Qwen3-4b
     - Llama-3.3-70b
     - Mistral-31-24b
     - Venice-Uncensored
     - Qwen3-235b

2. **Local File Purpose**:
   - This directory may contain local copies of configuration files **for debugging purposes only**
   - Local files are not used for active workflow execution
   - They serve as references for troubleshooting and analysis
   - Any modifications to workflows must be made on the server, not through local files

3. **Configuration Synchronization**:
   - Automated backups of server configurations are stored in `/a0/bitcore/backups/workflows/`
   - Backups are timestamped and verified with SHA-256 hashes
   - The backup system runs daily to ensure recent copies are available
   - Use the retrieval script in `research_engine/code/fetch_workflow_configs.py` to update local references

4. **Best Practices**:
   - Always verify the current configuration on the server before troubleshooting
   - Do not modify local configuration files expecting changes to take effect
   - Report any discrepancies between local references and server configurations
   - Use the backup system for historical configuration analysis

5. **Debugging Workflow**:
   ```bash
   # To retrieve current server configurations for debugging:
   python /a0/bitcore/workspace/research_engine/code/fetch_workflow_configs.py
   
   # To view available backup versions:
   ls -la /a0/bitcore/backups/workflows/
   
   # To compare local reference with server (requires authentication):
   python /a0/bitcore/workspace/research_engine/code/compare_configs.py --local <file> --remote <workflow_id>
   ```

### Important Notes

- **No Active Configurations**: This directory intentionally does not contain active configuration files to prevent confusion and configuration drift
- **Server as Source of Truth**: The n8n server is the single source of truth for all workflow configurations
- **Error-Prone Endpoint**: Note that the Llama-3.3-70b endpoint is currently marked as error-prone; monitor its performance and consider fallback strategies
- **Backup Integrity**: All backups include SHA-256 hashes for integrity verification; validate hashes before using backup files

This documentation ensures clarity about the configuration management strategy, preventing misunderstandings about where workflows are actually hosted and how local files should be used for debugging purposes.