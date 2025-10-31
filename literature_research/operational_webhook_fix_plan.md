# WEBHOOK CONFIGURATION FIX PLAN

## OBJECTIVE
Fix all neuron webhook configurations to ensure they accept POST requests, use correct JSON paths, and process payloads with 'researchTopic' field.

## CONSTRAINTS
- Do not modify core system files (/python/, /opt/venv/, etc.)
- Work only in /a0/projects/n8n/ for n8n workflows
- Use sequential execution with delays
- Validate each step before proceeding

## CONSTRUCT
1. Retrieve current n8n workflow configurations
2. Identify all neuron webhook nodes
3. Verify POST method is enabled
4. Correct JSON path from '$json.researchTopic' to '$json["Initialize Parameters"].json.researchTopic'
5. Ensure 'researchTopic' field is used in payload (not 'researchterm')
6. Implement staggered execution script with model-size-based delays

## VALIDATION
- Test each webhook individually after fix
- Verify response contains non-null data
- Confirm no 404 or 504 errors
- Validate JSON structure matches expectations

## DOCUMENTATION
- Update ARCHITECTURE.md with corrected workflow details
- Log all changes in implementation_log.md
- Save test results in /a0/projects/n8n/tests/

## EXECUTION PROTOCOL
- One webhook fixed and tested at a time
- 30 second delay between tests for small models, 60 seconds for large
- Immediate rollback if error occurs
- Full validation after each change

## ASSUMPTIONS TO VALIDATE
- User confirms this plan is approved before execution
- n8n service is running and accessible
- Webhook URLs are correctly configured in workflow files

This plan notarized on 2025-10-06 02:04:56-04:00
