import json
import sys
import os

def generate_html(json_data, output_file):
    sentences = json_data.get('sentences', [])
    groups = json_data.get('groups', [])
    
    # Map sentence index to sentence object for easy lookup
    sentence_map = {s['index']: s for s in sentences}
    
    html_content = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Sentence Grouping Report</title>",
        "    <style>",
        "        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #f4f7f6; }",
        "        h1 { color: #2c3e50; text-align: center; }",
        "        .group { background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; overflow: hidden; border: 1px solid #e0e0e0; }",
        "        .group-header { background: #3498db; color: #fff; padding: 15px 20px; margin: 0; font-size: 1.25rem; }",
        "        .group-labels { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 5px; }",
        "        .label-tag { background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: normal; }",
        "        .sentences-list { padding: 0; margin: 0; list-style: none; }",
        "        .sentence-item { padding: 15px 20px; border-bottom: 1px solid #eee; transition: background-color 0.2s; }",
        "        .sentence-item:last-child { border-bottom: none; }",
        "        .sentence-item:hover { background-color: #f9f9f9; }",
        "        .sentence-index { font-weight: bold; color: #7f8c8d; margin-right: 10px; font-size: 0.9rem; }",
        "        .sentence-text { display: inline; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Sentence Grouping Report</h1>"
    ]
    
    for group in groups:
        labels = group.get('label', [])
        label_text = " > ".join(labels)
        
        html_content.append("    <div class='group'>")
        html_content.append("        <div class='group-header'>")
        html_content.append(f"            <div>{label_text}</div>")
        html_content.append("            <div class='group-labels'>")
        for label in labels:
            html_content.append(f"                <span class='label-tag'>{label}</span>")
        html_content.append("            </div>")
        html_content.append("        </div>")
        html_content.append("        <ul class='sentences-list'>")
        
        for range_item in group.get('ranges', []):
            start = range_item['start']
            end = range_item['end']
            
            for i in range(start, end + 1):
                if i in sentence_map:
                    sent = sentence_map[i]
                    html_content.append("            <li class='sentence-item'>")
                    html_content.append(f"                <span class='sentence-index'>#{sent['index']}</span>")
                    html_content.append(f"                <span class='sentence-text'>{sent['text']}</span>")
                    html_content.append("            </li>")
        
        html_content.append("        </ul>")
        html_content.append("    </div>")
    
    html_content.extend([
        "</body>",
        "</html>"
    ])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <input_json> [output_html]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "report.html"
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)
        
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        generate_html(data, output_file)
        print(f"Report successfully generated: {output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)
