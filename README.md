# Matías Fernández Lakatos - Personal Portfolio 🚀

![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?logo=bootstrap&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

Welcome to my personal portfolio and project showcase! This repository hosts my GitHub Pages website featuring my academic background, research publications, and technical projects.

## 🌐 Live Website

Visit my portfolio at: **[mflakatos.github.io](https://mflakatos.github.io)**

## 👨‍🔬 About Me

I'm a **Senior Neuromorphic Researcher at Gradiant** (Vigo, Spain), leading applied research on **neuromorphic computing** since July 2026:

- **Third-generation neural networks** (Spiking Neural Networks) and synaptic plasticity
- **Event-based vision** with neuromorphic sensors (DVS)
- **SNN-to-FPGA compilation** and neuromorphic hardware; links with photonic architectures
- **Explainable AI (xAI)** for SNN-based models
- **Energy-efficient AI on the edge** — autonomous vehicles, manufacturing, healthcare use cases

Previously **Research Engineer at Gradiant** (2025 – Jun 2026), building intelligent solutions for early threat detection in corporate cybersecurity:

- **Unsupervised Machine Learning** (β-Variational Autoencoders, Autoencoders, tree-based methods)
- **Real-time Behavioral Analysis** on large-scale data
- **Explainable AI (XAI)** tools and integration
- **Scalable architectures** with Kafka, Spark, and Airflow

### 🎓 Academic Background

- **PhD in Optics** - Universidad de la República, Uruguay (2019-2024)
- **MSc in Quantum Chromodynamics** - Universidad de la República, Uruguay (2016-2018)
- **MSc in Big Data Analytics** - USC, Spain (2024-2025)

## 🔬 Research & Publications

### Featured Publications

#### As Lead Author
- **[SPIE 2024]** - Integration with one partial derivative applied to quantitative phase imaging
- **[Optics and Lasers in Engineering 2023]** - Scopus indexed research
- **[Optik 2022]** - Scopus indexed research
- **[International Journal of Modern Physics A 2019]** - Scopus indexed research

#### As Co-author
- **[Optics & Laser Technology 2025]** - Recent collaboration
- **[IOP Science 2024]** - Measurement Science and Technology

## 💻 Featured Projects

### 🖼️ Computer Vision & Image Processing
- **Digital Image Processing Pipeline** - Advanced techniques for phase object visualization and characterization
- **Multi-Camera Object Tracking** - Color-based centroid tracking system with multiple camera support
- **Range Detection System** - Computer vision application for distance measurement

### 📊 Financial Analytics
- **Stock Market Analysis Tools** - Python-based financial data analysis using Yahoo Finance API
- **Algorithmic Trading Backtesting** - SMA (Simple Moving Average) strategy implementation

### 🗂️ Media Organization Tools
- **Smart Media Organizer** - Automated image and video organization with metadata extraction
- **GUI-based File Management** - User-friendly interface for bulk media operations

## 🛠️ Technical Stack


### Machine Learning & Data Science
- **Deep Learning**: PyTorch, TensorFlow, Keras
- **Computer Vision**: OpenCV, scikit-image, PIL
- **Data Analysis**: pandas, NumPy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Big Data**: Apache Spark, Kafka, Airflow


## 📈 Current Focus Areas

- **Neuromorphic Computing**: Spiking Neural Networks, synaptic plasticity, neuromorphic hardware (FPGA, photonics)
- **Event-based Vision**: DVS sensors and their integration into real-world applications
- **Explainable AI**: xAI techniques for SNN-based models
- **Edge AI**: Energy-efficient, massively parallel AI systems

## 🤝 Connect With Me

- 📧 **Professional Contact**: Available in CV documents
- 🌐 **Portfolio**: [mflakatos.github.io](https://mflakatos.github.io)
- 📚 **Publications**: Links available on the main website
- 💼 **Current Role**: Senior Neuromorphic Researcher at Gradiant

## 🔄 How This Site Is Organized (and How to Update It)

### Repository structure

| Path | What it is |
|---|---|
| `index.html` | The whole web page (single file: styles + content) |
| `documents/` | All CVs: 4 profiles × 2 languages × 2 formats (16 files) |
| `documents/cv_english.md` / `cv_espanol.md` | **General** CV (markdown = source of truth) |
| `documents/cv_machine_learning_*.md` | ML-oriented CV |
| `documents/cv_computer_vision_*.md` | Computer-vision-oriented CV |
| `documents/cv_data_analyst_*.md` | Data-analyst-oriented CV |
| `imgs/`, `videos/`, `js/`, `estilos/` | Page assets |

### When there is job news (new role, project, certification…)

1. **Update the 8 markdown CVs** in `documents/` (both languages). The markdown files are the source of truth; PDFs are generated from them.
2. **Regenerate the 8 PDFs** from the markdown (or ask Claude to do it — keep filenames identical so the page links keep working).
3. **Update `index.html`** — the sections that mention your role, in order of appearance:
   - `<title>` tag
   - **Hero**: role chips, bio paragraph, stats
   - **What I Do**: domain cards
   - **Experience**: add a new timeline entry on top
   - **Technical Skills**: add new skills/pills
   - **Certifications**: add new certs
   - **Curriculum Vitae**: only changes if CV filenames change
4. **Update this README** — "About Me" and "Current Focus Areas".
5. Commit and push — GitHub Pages redeploys automatically in ~1 minute:

   ```bash
   git add . && git commit -m "Update role/CVs" && git push
   ```

## 📝 License

This portfolio website and associated projects are for educational and professional showcase purposes. Individual projects may have their own licensing terms.

---

<div align="center">

**"Journey before destination"** - *My philosophy for continuous learning and creative exploration*

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=MFLakatos.MFernandezLakatos.github.io)

</div>