
import json
import os
import re
from datetime import datetime

from workspace.config import cfg, get_logger

logger = get_logger(__name__)


def extract_key_findings(response_text):
    """Extract key findings from a model's response"""
    # Look for key sections in the response
    findings = []

    # Split response into sentences and look for important statements
    sentences = re.split(r'[.!?]+', response_text)

    # Look for sentences with high information density
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Ignore very short sentences
            # Look for sentences with technical terms or specific claims
            if any(term in sentence.lower() for term in ['hidden', 'latent', 'feature', 'state', 'pattern', 'model', 'hmm', 'markov', 'k-mer', 'entropy', 'surprisal', 'developability', 'antibody', 'sequence']):
                # Score based on information density
                words = sentence.split()
                if len(words) > 5:  # Not too short
                    # Simple scoring based on technical terms
                    score = sum(1 for term in ['hidden', 'latent', 'feature', 'state', 'pattern', 'model', 'hmm', 'markov', 'k-mer', 'entropy', 'surprisal', 'developability', 'antibody', 'sequence', 'cd', 'cdr', 'framework', 'variable', 'constant', 'domain'] if term in sentence.lower())
                    if score > 0:
                        findings.append({
                            "text": sentence,
                            "score": score,
                            "length": len(words)
                        })

    # Sort by score and length
    findings.sort(key=lambda x: (x["score"], x["length"]), reverse=True)

    return findings[:10]  # Return top 10 findings

def analyze_convergence(results_file):
    """Analyze convergence across models to identify high-confidence signals

    Args:
        results_file: path to JSON file with 'results' keyed by model name
    Returns:
        path to output file or None on failure
    """

    # Setup logging
    try:
        os.makedirs(os.path.dirname(cfg.LOG_FILE), exist_ok=True)
    except Exception:
        pass

    # Load the results
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Extract key findings from each model's response
    model_findings = {}

    for model_name, result in data["results"].items():
        if result.get("status") == "success":
            try:
                # Extract the response content
                response_content = result["response"]["choices"][0]["message"]["content"]
                model_findings[model_name] = extract_key_findings(response_content)
            except (KeyError, IndexError) as e:
                logger.error("Error extracting findings from %s: %s", model_name, e)
                model_findings[model_name] = []
        else:
            logger.error("Model %s failed: %s", model_name, result.get("message", "Unknown error"))
            model_findings[model_name] = []

    # Identify convergent findings (appear in multiple models)
    all_findings = []
    for model_name, findings in model_findings.items():
        for finding in findings:
            # Normalize the finding text for comparison
            normalized_text = re.sub(r'[^a-zA-Z0-9]', '', finding["text"]).lower()
            all_findings.append({
                "model": model_name,
                "text": finding["text"],
                "normalized": normalized_text,
                "score": finding["score"],
                "length": finding["length"]
            })

    # Group findings by normalized text
    finding_groups = {}
    for finding in all_findings:
        if finding["normalized"] not in finding_groups:
            finding_groups[finding["normalized"]] = []
        finding_groups[finding["normalized"]].append(finding)

    # Identify convergent findings (appear in multiple models)
    convergent_findings = []
    for normalized_text, group in finding_groups.items():
        if len(group) > 1:  # Appears in multiple models
            # Calculate consensus score
            models = [f["model"] for f in group]
            unique_models = len(set(models))
            total_score = sum(f["score"] for f in group)

            # Select the best version of the finding
            best_finding = max(group, key=lambda x: (x["score"], x["length"]))

            convergent_findings.append({
                "text": best_finding["text"],
                "models": models,
                "unique_models": unique_models,
                "total_score": total_score,
                "average_score": total_score / len(group)
            })

    # Sort by consensus (number of unique models) and total score
    convergent_findings.sort(key=lambda x: (x["unique_models"], x["total_score"]), reverse=True)

    # Create validated findings with confidence levels
    validated_findings = []
    for finding in convergent_findings:
        # Assign confidence level based on consensus
        if finding["unique_models"] >= 5:
            confidence = "high"
        elif finding["unique_models"] >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        validated_findings.append({
            "finding": finding["text"],
            "supporting_models": finding["models"],
            "model_count": finding["unique_models"],
            "confidence": confidence,
            "evidence_score": finding["total_score"]
        })

    # Save validated findings to cfg.RESULTS_BASE_DIR/validated_findings or fallback to tmp
    validated_dir = os.path.join(cfg.RESULTS_BASE_DIR, "validated_findings")
    try:
        os.makedirs(validated_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(validated_dir, f"validated_findings_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump({
                "analysis_time": datetime.now().isoformat(),
                "input_file": results_file,
                "total_convergent_findings": len(validated_findings),
                "high_confidence_findings": len([f for f in validated_findings if f["confidence"] == "high"]),
                "medium_confidence_findings": len([f for f in validated_findings if f["confidence"] == "medium"]),
                "low_confidence_findings": len([f for f in validated_findings if f["confidence"] == "low"]),
                "findings": validated_findings
            }, f, indent=2)
        logger.info(
            "Validation complete. %s convergent findings identified. Results saved to %s",
            len(validated_findings),
            output_file,
        )
        return output_file
    except Exception as e:
        logger.error("Error saving validated findings to %s: %s", validated_dir, e)
        try:
            import tempfile
            tmpdir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fb = os.path.join(tmpdir, f"validated_findings_{timestamp}.json")
            with open(fb, 'w') as f:
                json.dump({
                    "analysis_time": datetime.now().isoformat(),
                    "input_file": results_file,
                    "total_convergent_findings": len(validated_findings),
                    "high_confidence_findings": len([f for f in validated_findings if f["confidence"] == "high"]),
                    "medium_confidence_findings": len([f for f in validated_findings if f["confidence"] == "medium"]),
                    "low_confidence_findings": len([f for f in validated_findings if f["confidence"] == "low"]),
                    "findings": validated_findings
                }, f, indent=2)
            logger.info("Validation complete. Results saved to fallback %s", fb)
            return fb
        except Exception as e2:
            logger.error("Final fallback failed: %s", e2)
            return None


def run_validation(results_file=None):
    """Convenience runner for analyze_convergence.

    If results_file is None, attempt to locate a file in cfg.RESULTS_RAW_DIR.
    """
    if results_file is None:
        try:
            raw_dir = cfg.RESULTS_RAW_DIR
            candidates = []
            for fn in os.listdir(raw_dir):
                if fn.startswith('research_results_') and fn.endswith('.json'):
                    candidates.append(os.path.join(raw_dir, fn))
            if not candidates:
                logger.warning("No candidate results files in %s", raw_dir)
                return None
            # pick latest by modification time
            results_file = max(candidates, key=os.path.getmtime)
        except Exception as e:
            logger.error("Error locating results_file in %s: %s", cfg.RESULTS_RAW_DIR, e)
            return None

    return analyze_convergence(results_file)


if __name__ == '__main__':
    # When run as script, attempt to validate latest results
    run_validation()
