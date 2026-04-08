# multimodal-rag

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pull REAL-MM-RAG data

See scripts/get_data.py for the helper to pull the REAL-MM-RAG dataset.

- Default datasets from hf: `ibm-research/REAL-MM-RAG_FinReport`, `ibm-research/REAL-MM-RAG_TechReport`, `ibm-research/REAL-MM-RAG_TechSlides`

Test run download (only 10 examples per split):

```bash
python3 scripts/get_data.py --out_dir data/raw --max_examples 10
```

Full download (all default datasets):

```bash
python3 scripts/get_data.py --out_dir data/raw
```

By default the script saves page images and writes two metadata files per split:

- `data/raw/<dataset_short>/<split>/pages/` — PNG page images (one file per page)
- `data/raw/<dataset_short>/<split>/pages.jsonl` — page-level metadata (image path, doc id, page number, optional OCR)
- `data/raw/<dataset_short>/<split>/qas.jsonl` — QA triples linking questions/answers to page images

**Optional OCR**:
To download OCR text saved alongside images (for baselines), install `tesseract` binary from homebrew.

Homebrew (macOS):

```bash
brew install tesseract
```

Amazon Linux Install:

```bash
# install system dependencies
sudo dnf update -y
sudo dnf groupinstall "Development Tools" -y
sudo dnf install wget clang gcc-c++ libjpeg-devel libpng-devel libtiff-devel zlib-devel autoconf automake libtool -y

# build Leptonica (dependency of Tesseract)
cd /tmp
wget https://github.com/DanBloomberg/leptonica/releases/download/1.84.1/leptonica-1.84.1.tar.gz
tar -xvf leptonica-1.84.1.tar.gz
cd leptonica-1.84.1
./configure
make -j$(nproc)
sudo make install

# configure local paths for the build process
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
export LDFLAGS="-L/usr/local/lib"
export CPPFLAGS="-I/usr/local/include"

# build Tesseract 5.3.4
cd /tmp
wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.3.4.tar.gz
tar -xvf 5.3.4.tar.gz
cd tesseract-5.3.4
./autogen.sh
./configure
make -j$(nproc)
sudo make install

# link libraries and download English data
sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/usr-local.conf'
sudo ldconfig
sudo mkdir -p /usr/local/share/tessdata
sudo wget -P /usr/local/share/tessdata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

# make environment variables persistent for all future sessions
cat << 'EOF' >> ~/.bashrc

# Tesseract & Leptonica Paths (GenAI Project)
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
export TESSDATA_PREFIX=/usr/local/share/tessdata
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
EOF

# apply changes to current session
source ~/.bashrc
tesseract --version
```

Pull and extract metadata with OCR (10 examples):

```bash
python3 scripts/get_data.py --out_dir data/raw --run_ocr --max_examples 10
```

Full data:

```bash
python3 scripts/get_data.py --out_dir data/raw --run_ocr
```
