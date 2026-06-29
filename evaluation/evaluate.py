import json
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests

def run_evaluation_suite(test_cases, progress_callback=None, use_reranker=True):
    """
    Runs the evaluation suite over a list of test cases.
    progress_callback: A function that takes (current_index, total_questions, status_text)
    """
    total_questions = len(test_cases)
    if total_questions == 0:
        return None
        
    correct_documents_retrieved = 0
    total_retrieval_time = 0.0
    total_response_time = 0.0
    total_confidence = 0.0
    all_results = []
    
    for i, test in enumerate(test_cases):
        if progress_callback:
            progress_callback(i, total_questions, f"Running evaluation: Question {i+1} of {total_questions}...")
            
        q = test["question"]
        expected_keywords = test.get("expected_answer_keywords", [])
        
        start_time = time.time()
        # Generate answer from our pipeline via HTTP API
        try:
            payload = {"question": q, "session_id": None, "use_reranker": use_reranker}
            # Hardcoded to local dev server for now; in prod this could be parameterized
            res = requests.post("http://127.0.0.1:8000/api/v1/ask", json=payload)
            res.raise_for_status()
            data = res.json()
            
            answer = data.get("answer", "")
            retrieval_time = data.get("retrieval_time", 0.0)
            context = data.get("context", "")
            confidence = data.get("confidence", 0.0)
            sources = data.get("sources", [])
        except Exception as e:
            answer = f"Error: {e}"
            retrieval_time = 0.0
            context = ""
            confidence = 0.0
            sources = []
            
        end_time = time.time()
        response_time = end_time - start_time
        total_retrieval_time += retrieval_time
        total_response_time += response_time
        total_confidence += confidence
        
        # Check correctness of retrieved document: 
        is_retrieved_correctly = False
        if not expected_keywords:
            is_retrieved_correctly = True
        else:
            is_retrieved_correctly = any(str(kw).lower() in context.lower() for kw in expected_keywords)
            
        if is_retrieved_correctly:
            correct_documents_retrieved += 1
            
        # Keep track of answer correctness
        is_answer_correct = False
        if not expected_keywords:
            is_answer_correct = True
        else:
            is_answer_correct = any(str(kw).lower() in answer.lower() for kw in expected_keywords)
            
        all_results.append({
            "question": q,
            "expected": expected_keywords,
            "actual": answer,
            "confidence": confidence,
            "sources": sources,
            "is_correct": is_answer_correct
        })
        
    retrieval_accuracy = (correct_documents_retrieved / total_questions) * 100 if total_questions > 0 else 0
    avg_retrieval_time = total_retrieval_time / total_questions if total_questions > 0 else 0
    avg_response_time = total_response_time / total_questions if total_questions > 0 else 0
    avg_confidence = total_confidence / total_questions if total_questions > 0 else 0
    
    return {
        "total_questions": total_questions,
        "correct_documents_retrieved": correct_documents_retrieved,
        "retrieval_accuracy": retrieval_accuracy,
        "avg_retrieval_time": avg_retrieval_time,
        "avg_response_time": avg_response_time,
        "avg_confidence": avg_confidence,
        "all_results": all_results
    }

def run_evaluation():
    print("=== Starting RAG Pipeline Evaluation ===\n")
    
    questions_path = os.path.join(os.path.dirname(__file__), 'test_questions.json')
    try:
        with open(questions_path, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("Error: test_questions.json not found.")
        return

    def print_progress(i, total, text):
        print(text)
        
    results = run_evaluation_suite(test_cases, progress_callback=print_progress)
    
    if not results:
        print("No test questions found.")
        return
        
    print("\n=== Evaluation Complete ===")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Retrieval Accuracy: {results['retrieval_accuracy']:.1f}%")
    print(f"Avg Retrieval Time: {results['avg_retrieval_time']:.4f}s")
    print(f"Avg Response Time: {results['avg_response_time']:.2f}s")
    print(f"Avg Confidence: {results['avg_confidence']:.2f}")

def render_evaluation_ui(st, project_root):
    questions_path = os.path.join(project_root, 'evaluation', 'test_questions.json')
    try:
        with open(questions_path, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        st.error("Error: test_questions.json not found.")
        st.stop()

    if not test_cases:
        st.warning("No test questions found.")
        st.stop()
        
    st.markdown("### Running Evaluation (Without Reranking)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(i, total, text):
        status_text.text(text)
        progress_bar.progress((i + 1) / total)
        
    results_no_rerank = run_evaluation_suite(test_cases, progress_callback=update_progress, use_reranker=False)
    
    status_text.empty()
    progress_bar.empty()

    st.markdown("### Running Evaluation (With Reranking)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
        
    results_with_rerank = run_evaluation_suite(test_cases, progress_callback=update_progress, use_reranker=True)
    
    status_text.empty()
    progress_bar.empty()
    
    if not results_with_rerank or not results_no_rerank:
        st.stop()
    
    # Display metrics in a markdown table comparing both
    st.subheader("Results Summary (Comparison)")
    
    def get_diff(val1, val2, is_percentage=False, is_time=False, invert_good=False):
        diff = val2 - val1
        if diff == 0:
            return ""
        
        sign = "+" if diff > 0 else ""
        
        # For time, lower is better. For accuracy/confidence, higher is better
        is_good = (diff < 0) if invert_good else (diff > 0)
        color = "green" if is_good else "red"
        
        if is_percentage:
            return f" <span style='color:{color}'>({sign}{diff:.0f}%)</span>"
        elif is_time:
            return f" <span style='color:{color}'>({sign}{diff:.4f}s)</span>"
        else:
            return f" <span style='color:{color}'>({sign}{diff:.2f})</span>"
    
    table_md = f"""
| Metric | Without Reranking | With Reranking | Change |
| --- | --- | --- | --- |
| Questions Tested | {results_no_rerank['total_questions']} | {results_with_rerank['total_questions']} | |
| Correct Document Retrieved | {results_no_rerank['correct_documents_retrieved']} | {results_with_rerank['correct_documents_retrieved']} | {get_diff(results_no_rerank['correct_documents_retrieved'], results_with_rerank['correct_documents_retrieved'], invert_good=False)} |
| Retrieval Accuracy | {results_no_rerank['retrieval_accuracy']:.0f}% | {results_with_rerank['retrieval_accuracy']:.0f}% | {get_diff(results_no_rerank['retrieval_accuracy'], results_with_rerank['retrieval_accuracy'], is_percentage=True)} |
| Average Retrieval Time | {results_no_rerank['avg_retrieval_time']:.4f} s | {results_with_rerank['avg_retrieval_time']:.4f} s | {get_diff(results_no_rerank['avg_retrieval_time'], results_with_rerank['avg_retrieval_time'], is_time=True, invert_good=True)} |
| Average Response Time | {results_no_rerank['avg_response_time']:.2f} s | {results_with_rerank['avg_response_time']:.2f} s | {get_diff(results_no_rerank['avg_response_time'], results_with_rerank['avg_response_time'], is_time=True, invert_good=True)} |
| Average Confidence | {results_no_rerank['avg_confidence']:.2f} | {results_with_rerank['avg_confidence']:.2f} | {get_diff(results_no_rerank['avg_confidence'], results_with_rerank['avg_confidence'], invert_good=False)} |
"""
    st.markdown(table_md, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Question Details (With Reranking)")
    for res in results_with_rerank["all_results"]:
        # Give a visual indicator if generation failed
        status_icon = "✅" if res["is_correct"] else "❌"
        with st.expander(f"{status_icon} Question: {res['question']}"):
            st.markdown(f"**Confidence Score:** `{res['confidence']:.2f}`")
            st.markdown(f"**Expected Keywords:** {', '.join(res['expected'])}")
            st.markdown(f"**Actual Answer:**\n{res['actual']}")
            
            if res['sources']:
                st.markdown("**Retrieved Sources:**")
                for src in res['sources']:
                    # Citation has document and optionally page
                    source_name = getattr(src, 'document', 'Unknown')
                    if source_name == 'Unknown' and isinstance(src, dict):
                        source_name = src.get('document', 'Unknown')
                        page = src.get('page', None)
                    else:
                        page = getattr(src, 'page', None)
                        
                    page_str = f" (Page {page})" if page else ""
                    st.markdown(f"- 📄 {source_name}{page_str}")
            else:
                st.markdown("**Retrieved Sources:** None")


if __name__ == "__main__":
    run_evaluation()
