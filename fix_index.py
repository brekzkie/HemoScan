import os

def fix():
    html_path = r"C:\Users\User\Documents\HemoScan\index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    start_marker = "        const HemoScanLoader = ({ size = '11rem' }) => ("
    end_marker = '        const API_URL = "http://localhost:8000";'

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        print("Found markers!")
        new_loader = """        const HemoScanLoader = ({ size = '11rem' }) => (
            <div className="hemoscan-loader-container" style={{ width: size, height: size }}>
                <video 
                    src="loading.mp4" 
                    autoPlay 
                    loop 
                    muted 
                    playsInline 
                    style={{ 
                        width: '100%', 
                        height: '100%', 
                        objectFit: 'cover', 
                        borderRadius: '2rem',
                        border: '2px solid rgba(149, 37, 40, 0.15)'
                    }} 
                />
            </div>
        );

"""
        new_content = content[:start_idx] + new_loader + content[end_idx:]
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Successfully replaced broken loader!")
    else:
        print("Error: markers not found")
        # Try a more relaxed search if spaces differ
        if "const HemoScanLoader = " in content:
            print("Found component name but not exact signature")

if __name__ == "__main__":
    fix()
