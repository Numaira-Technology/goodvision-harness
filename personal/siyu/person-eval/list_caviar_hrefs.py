import re
import urllib.request

url = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/"
html = urllib.request.urlopen(url, timeout=30).read().decode("latin-1", errors="replace")
hrefs = sorted(set(re.findall(r'href=["\']([^"\']+\.mpg)["\']', html, re.I)))
print("count", len(hrefs))
for item in hrefs:
    print(item)
