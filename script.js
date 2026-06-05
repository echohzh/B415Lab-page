const labInfo = {
  name: "新疆大学语音感知与智能计算实验室",
  subtitle: "语音语言与音频信号处理",
  office: "新疆大学博达校区信息楼 B405 & B415",
  emails: ["zhhuang@xju.edu.cn", "echohzh@163.com"],
  source: "https://it.xju.edu.cn/info/1150/2085.htm"
};

const directions = [
  {
    id: "enhancement",
    name: "语音增强",
    english: "Speech Enhancement",
    accent: "#087c70",
    summary:
      "围绕复杂的现实环境与多语言场景，在低信噪比条件下恢复出语音内容，提升语音信号的感知质量、可懂度和清晰度。",
    projects: [
      {
        id: "dytswig",
        kind: "Demo + 开源代码",
        title: "DyTSwiG-Mamba",
        description:
          "基于双分支相位预测的无层归一化 CNN-Mamba 语音增强网络，提供英文与中文语音增强对比样例。",
        result: "支持训练、推理、客观指标评测和中英文增强试听。",
        tags: ["CNN-Mamba", "Dual-branch Phase Prediction", "Input-biased Dynamic Tanh", "Mandarin Enhancement"],
        year: "2025",
        repoPath: "assets/DyTSwiG-SE-main/",
        links: [
          { label: "Demo", url: "assets/DyTSwiG-SE-main/index.html" },
          { label: "Code", url: "https://github.com/Yj-Xiong/DyTSwiG-SE" },
        ],
        samples: [
          {
            title: "p257_020",
            text: "VoiceBank+DEMAND 英文样例，对比干净语音、带噪输入、DBPD 输出以及 mapping/masking 分支。",
            tracks: [
              { label: "Clean (ref)", src: "assets/DyTSwiG-SE-main/EnglishDemos/p257_020/p257_020 clean.wav" },
              { label: "Noisy (input)", src: "assets/DyTSwiG-SE-main/EnglishDemos/p257_020/p257_020 noisy.wav" },
              { label: "DBPD (ours)", src: "assets/DyTSwiG-SE-main/EnglishDemos/p257_020/p257_020 dbpd.wav" },
              { label: "Mapping branch", src: "assets/DyTSwiG-SE-main/EnglishDemos/p257_020/p257_020 mapping.wav" },
              { label: "Masking branch", src: "assets/DyTSwiG-SE-main/EnglishDemos/p257_020/p257_020 masking.wav" }
            ]
          },
          {
            title: "D8_956",
            text: "THCHS-30 普通话样例，对比带噪输入、干净语音、DyTSwiG-Mamba 以及 SEMamba、CMGAN、PrimeK-Net 输出。",
            tracks: [
              { label: "Noisy (input)", src: "assets/DyTSwiG-SE-main/MandarinDemos/D8_956/D8_956 noisy.wav" },
              { label: "Reference (clean)", src: "assets/DyTSwiG-SE-main/MandarinDemos/D8_956/D8_956 reference.wav" },
              { label: "DyTSwiG-Mamba (ours)", src: "assets/DyTSwiG-SE-main/MandarinDemos/D8_956/D8_956 ours.wav" },
              { label: "SEMamba", src: "assets/DyTSwiG-SE-main/MandarinDemos/D8_956/D8_956 SEmamba.wav" },
              { label: "CMGAN", src: "assets/DyTSwiG-SE-main/MandarinDemos/D8_956/D8_956 cmgan.wav" },
              { label: "PrimeK-Net", src: "assets/DyTSwiG-SE-main/MandarinDemos/D8_956/D8_956 primek.wav" }
            ]
          },
          {
            title: "D21_837",
            text: "THCHS-30 普通话样例，对比带噪输入、干净语音、DyTSwiG-Mamba 以及 SEMamba、CMGAN、PrimeK-Net 输出。",
            tracks: [
              { label: "Noisy (input)", src: "assets/DyTSwiG-SE-main/MandarinDemos/D21_837/Noisy-D21_837.wav" },
              { label: "Reference (clean)", src: "assets/DyTSwiG-SE-main/MandarinDemos/D21_837/D21_837.wav" },
              { label: "DyTSwiG-Mamba (ours)", src: "assets/DyTSwiG-SE-main/MandarinDemos/D21_837/Ours-D21_837.wav" },
              { label: "SEMamba", src: "assets/DyTSwiG-SE-main/MandarinDemos/D21_837/SEMamba-D21_837.wav" },
              { label: "CMGAN", src: "assets/DyTSwiG-SE-main/MandarinDemos/D21_837/CMGAN-D21_837.wav" },
              { label: "PrimeK-Net", src: "assets/DyTSwiG-SE-main/MandarinDemos/D21_837/Primek-D21_837.wav" }
            ]
          }
        ]
      },
      {
        id: "tdsenet",
        kind: "Demo + 开源代码",
        title: "Td-SENet",
        description:
          "面向跨语言和低信噪比场景的三重解码器语音增强模型，强化局部特征建模与高频信息保留。",
        result: "涵盖训练、推理及评估，提供汉语对比试听。",
        tags: ["Triple-decoder", "Local-augmented Conformer", "High-frequency Information","Cross-lingual", "Low-SNR", ],
        year: "2025",
        repoPath: "assets/TdSENet-main/",
        links: [
          { label: "Demo", url: "assets/TdSENet-main/index.html" },
          { label: "Code", url: "https://github.com/Yj-Xiong/TdSENet" },
        ],
        samples: [
          {
            title: "Mandarin sample B7_278",
            text: "-9dB信噪比下的普通话语音增强效果，对比带噪语音（Noisy）、干净语音（Clean） 与 Td-SENet 输出。",
            tracks: [
              { label: "Noisy -9 dB", src: "assets/TdSENet-main/B7_278/B7_278_-9dB.wav" },
              { label: "Clean", src: "assets/TdSENet-main/B7_278/B7_278 clean.wav" },
              { label: "Td-SENet", src: "assets/TdSENet-main/B7_278/B7_278 tdnet.wav" }
            ]
          },
          {
            title: "Mandarin sample A4_204",
            text: "极端噪声环境下增强对比样例。",
            tracks: [
              { label: "Noisy -9 dB", src: "assets/TdSENet-main/A4_204/A4_204_-9dB.wav" },
              { label: "Clean", src: "assets/TdSENet-main/A4_204/A4_204 clean.wav" },
              { label: "Td-SENet", src: "assets/TdSENet-main/A4_204/A4_204 tdnet.wav" }
            ]
          }
        ]
      },
      {
        id: "mcmgan",
        kind: "开源代码",
        title: "M-CMGAN",
        description:
          "在 CMGAN 框架中引入 Time-Mamba-Frequency-Conformer、通道与频率全局调制模块，较CMGAN可以更高效地实现单通道语音增强，同时保障增强效果。",
        result: "支持训练、推理和客观指标计算。",
        tags: ["CMGAN", "Mamba", "Conformer"],
        year: "2024",
        repoPath: "assets/M-CMGAN/",
        links: [
          { label: "Paper", url: "https://link.springer.com/chapter/10.1007/978-981-96-1045-7_2" },
          { label: "Code", url: "https://github.com/Yj-Xiong/M-CMGAN" },
        ]
      }
    ]
  },
  {
    id: "tts",
    name: "语音合成",
    english: "Speech Synthesis",
    accent: "#c4553e",
    summary:
      "聚焦 zero-shot TTS、语音转换、语码转换语音合成和低资源维吾尔语合成，提升自然度、可懂度和说话人相似度。",
    projects: [
      {
        id: "tvm",
        kind: "Demo 页面",
        title: "TV-MDiff",
        description:
          "A zero-shot text-to-speech and voice conversion system with Mamba-based diffusion model，覆盖 zero-shot TTS、zero-shot VC 与中文迁移样例。",
        result: "覆盖 VCTK 英文 TTS/VC、未标注数据训练和中文迁移试听。",
        tags: ["Zero-shot TTS", "Voice Conversion", "Mamba", "Diffusion"],
        year: "2025",
        repoPath: "assets/TV-MDiff-main/",
        links: [
          { label: "Paper", url: "https://ieeexplore.ieee.org/document/11227557" },
          { label: "Demo", url: "assets/TV-MDiff-main/index.html" }
        ],
        samples: [
          {
            title: "Zero-shot TTS sentence 1",
            text: "I know when my time is, it doesn't bother me.",
            tracks: [
              { label: "Reference", src: "assets/TV-MDiff-main/audio/ZS_TTS/sentence1/p305_388_mic2.flac" },
              { label: "GT", src: "assets/TV-MDiff-main/audio/ZS_TTS/sentence1/p305_240_mic2.flac" },
              { label: "TV-MDiff", src: "assets/TV-MDiff-main/audio/ZS_TTS/sentence1/TTS_MDiff_568.wav" }
            ]
          },
          {
            title: "Transferability of ZS TTS",
            text: "在陈水扁担任台北市议员时，苏焕智曾担任扁助理。",
            tracks: [
              { label: "Reference", src: "assets/TV-MDiff-main/audio/additional/1/p276_349_mic2.flac" },
              { label: "YourTTS", src: "assets/TV-MDiff-main/audio/additional/1/TTS_yourtts_cn_68.wav" },
              { label: "TV-MDiff", src: "assets/TV-MDiff-main/audio/additional/1/TTS_MDiff_cn2_68.wav" }
            ]
          }
        ]
      },
      {
        id: "unitdiff",
        kind: "Demo 页面",
        title: "UnitDiff",
        description:
          "A unit-diffusion model for code-switching speech synthesis，使用 soft HuBERT unit 直接预测 clean mel-spectrogram，并通过语言标记提升可懂度。",
        result: "展示中英单语、跨语言和 code-switching 合成效果。",
        tags: ["Code-switching", "Unit Diffusion", "Soft HuBERT", "Speaker Control"],
        year: "2025",
        repoPath: "assets/unitdiff-main/",
        links: [
          { label: "Paper", url: "https://ieeexplore.ieee.org/document/10891773" },
          { label: "Demo", url: "assets/unitdiff-main/" }],
        samples: [
          {
            title: "Code-switching synthesis",
            text: "我今天在home超级happy. 你今天过得怎么样？Do you have any plan to go to 学校?",
            tracks: [
              { label: "YourTTS", src: "assets/unitdiff-main/zh/MOS/CN/CS/MOS_CS_CN_yourtts_text_01.wav" },
              { label: "PPG-model", src: "assets/unitdiff-main/zh/MOS/CN/CS/MOS_CS_CN_ppg_model_01.wav" },
              { label: "UnitDiff", src: "assets/unitdiff-main/zh/MOS/CN/CS/MOS_CS_CN_unitdiff_01.wav" }
            ]
          },
          {
            title: "Cross-lingual synthesis",
            text: "最终，中国男子乒乓球队获得此奖项。",
            tracks: [
              { label: "YourTTS", src: "assets/unitdiff-main/zh/MOS/EN/cross/MOS_Cross_EN_yourtts_text_02.wav" },
              { label: "PPG-model", src: "assets/unitdiff-main/zh/MOS/EN/cross/MOS_Cross_EN_ppg_model_02.wav" },
              { label: "UnitDiff", src: "assets/unitdiff-main/zh/MOS/EN/cross/MOS_Cross_EN_unitdiff_02.wav" }
            ]
          }
        ]
      },
      {
        id: "zcscdiff",
        kind: "Demo 页面",
        title: "ZCS-CDiff",
        description:
          "A zero-shot code-switching TTS system with Conformer-based diffusion model，面向未见说话人的语码转换语音合成。",
        result: "覆盖 intra-lingual、cross-lingual、code-switching 和消融对比。",
        tags: ["Zero-shot", "Code-switching", "Conformer", "Diffusion"],
        year: "2025",
        repoPath: "assets/zcs-cdiff-main/",
        links: [
          { label: "Paper", url: "https://ieeexplore.ieee.org/document/10889531" },
          { label: "Demo", url: "assets/zcs-cdiff-main/index.html" }],
        samples: [
          {
            title: "Zero-shot code-switching TTS",
            text: "最近的新闻报道引发了很多关于social justice的讨论，大家都在关注如何构建一个更加公平和公正的社会。",
            tracks: [
              { label: "Reference", src: "assets/zcs-cdiff-main/audio/cs/en/sentence1/p306_244_mic2.flac" },
              { label: "YourTTS", src: "assets/zcs-cdiff-main/audio/cs/en/sentence1/yourtts_CS_EN_937.wav" },
              { label: "ZCS-CDiff", src: "assets/zcs-cdiff-main/audio/cs/en/sentence1/CDiff_CS_EN_937.wav" }
            ]
          },
          {
            title: "Cross-lingual TTS",
            text: "冬天的雪景总是让人感到无比的宁静与祥和，仿佛世界都安静了下来。",
            tracks: [
              { label: "Reference", src: "assets/zcs-cdiff-main/audio/cross/en/sentence1/p294_086_mic2.flac" },
              { label: "FastSpeech 2", src: "assets/zcs-cdiff-main/audio/cross/en/sentence1/fs2_cross_EN_200.wav" },
              { label: "ZCS-CDiff", src: "assets/zcs-cdiff-main/audio/cross/en/sentence1/CDiff_cross_EN_200.wav" }
            ]
          }
        ]
      },
      {
        id: "uyg-six",
        kind: "Demo 页面",
        title: "维吾尔语六模型合成对比",
        description:
          "对 FastSpeech2、Grad-TTS、VITS1、VITS2、RAW、XLMR 六种维吾尔语语音合成模型进行音频对比。",
        result: "对比 6 类模型在 25 组维吾尔语文本上的合成表现。",
        tags: ["Uyghur", "FastSpeech2", "Grad-TTS", "VITS"],
        year: "2024",
        repoPath: "assets/Six-different-models-for-synthesizing-Uyghur-language-main/",
        links: [
          { label: "Demo", url: "assets/Six-different-models-for-synthesizing-Uyghur-language-main/index.html" },
        ],
        samples: [
          {
            title: "Uyghur synthesis comparison",
            text: "ئىككى خىل ئېلېمېنتنى ئارىلاشتۇرسا خىمىيىلىك رېئاكسىيە يۈز بېرىدۇ",
            tracks: [
              {
                label: "FastSpeech2",
                src: "assets/Six-different-models-for-synthesizing-Uyghur-language-main/chenke_ug_fs2_new_25/fs2_UG_FS2_textThuyg25.txt_ug_1.wav"
              },
              {
                label: "Grad-TTS",
                src: "assets/Six-different-models-for-synthesizing-Uyghur-language-main/chenke_ug_Grad-TTS_new_25/grad_UYG_textThuyg25.txt_ug_1.wav"
              },
              {
                label: "VITS2",
                src: "assets/Six-different-models-for-synthesizing-Uyghur-language-main/uyg_vits2_newug25_20240910/0.wav"
              }
            ]
          }
        ]
      },
      {
        id: "uyg-three",
        kind: "Demo 页面",
        title: "维吾尔语三模型 Zero-shot 合成",
        description:
          "对 FastSpeech2、YourTTS、MDDM 三种模型在 CN-UG、EN-UG、UG-UG 场景下的 zero-shot 维吾尔语合成效果进行对比。",
        result: "对比中文、英文、维吾尔语说话人条件下的跨语种合成效果。",
        tags: ["Uyghur", "Zero-shot", "MDDM", "YourTTS"],
        year: "2024",
        repoPath: "assets/Three-different-models-for-synthesizing-Uyghur-language-main/",
        links: [
          { label: "Demo", url: "assets/Three-different-models-for-synthesizing-Uyghur-language-main/index.html" },
        ],
        samples: [
          {
            title: "CN-UG zero-shot synthesis",
            text: "بۇ يىل ھۆل يېغىن ياخشى بولغانلىقتىن زىرائەتلەرنىڭ دېنى ناھايىتى توق بولدى.",
            tracks: [
              {
                label: "FastSpeech2",
                src: "assets/Three-different-models-for-synthesizing-Uyghur-language-main/zhy_CN-UG_fs2_gen/fs2_UG_textThuyg20.txt_gen_7.wav"
              },
              {
                label: "YourTTS",
                src: "assets/Three-different-models-for-synthesizing-Uyghur-language-main/zhy_CN-UG_yourtts_gen/yourtts_UG_textThuyg20.txt_gen_7.wav"
              },
              {
                label: "MDDM",
                src: "assets/Three-different-models-for-synthesizing-Uyghur-language-main/zhy_CN-UG_MDDM_gen/MDDM_UG_textThuyg20.txt_gen_7.wav"
              }
            ]
          }
        ]
      }
    ]
  }
];

const publications = [
  {
    year: "2025",
    title: "TV-MDiff: A Zero-Shot Text-To-Speech and Zero-Shot Voice Conversion System with Mamba-Based Diffusion Model",
    venue: "International Joint Conference on Neural Networks (IJCNN)",
    authors: "Huang Z, Chen K, Yan Y",
    link: "https://ieeexplore.ieee.org/document/11227557"
  },
  {
    year: "2025",
    title: "UnitDiff: A Unit-Diffusion Model for Code-Switching Speech Synthesis",
    venue: "IEEE Signal Processing Letters, 32:1051-1055",
    authors: "Chen K, Huang Z, He L, Yan Y",
    link: "https://ieeexplore.ieee.org/document/10891773"
  },
  {
    year: "2025",
    title: "ZCS-CDiff: A Zero-Shot Code-Switching TTS System with Conformer-based Diffusion Model",
    venue: "IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
    authors: "Chen K, Huang Z, He L, Yan Y",
    link: "https://ieeexplore.ieee.org/document/10889531"
  },
  {
    year: "2024",
    title: "Fast Sampling Based on Policy Gradient for Diffusion-Based Speech Enhancement",
    venue: "IEEE 14th International Symposium on Chinese Spoken Language Processing (ISCSLP), 576-580",
    authors: "Jiang Y, Huang Z",
    link: "https://doi.org/10.1109/iscslp63861.2024.10799970"
  },
  {
    year: "2024",
    title: "M-CMGAN: Attempting to Use Mamba on Speech Enhancement",
    venue: "National Conference on Man-Machine Speech Communication (NCMMSC), 15-27",
    authors: "Xiong Y, Huang Z",
    link: "https://link.springer.com/chapter/10.1007/978-981-96-1045-7_2"
  },
  {
    year: "2024",
    title: "Optimizing Uyghur Speech Synthesis by Combining Pretrained Cross-Lingual Model",
    venue: "ACM Transactions on Asian and Low-Resource Language Information Processing, 23(9):1-11",
    authors: "Lu K, Huang Z, Yin M, Chen K",
    link: "https://doi.org/10.1145/3675397"
  },
  {
    year: "2024",
    title: "CosDiff: Code-Switching TTS Model Based on A Multi-Task DDIM",
    venue: "IEEE International Conference on Multimedia and Expo (ICME)",
    authors: "Chen K, Huang Z, Lu K, Yan Y",
    link: "https://doi.org/10.1109/icme57554.2024.10687605"
  }
];

const projects = directions.flatMap((direction) =>
  direction.projects.map((project) => ({ ...project, direction }))
);

function renderDirectionIndex() {
  const holder = document.querySelector("#directionIndex");
  if (!holder) return;

  holder.innerHTML = directions
    .map(
      (direction) => `
        <a class="index-item" style="--accent: ${direction.accent}" href="#${direction.id}">
          <i class="index-dot" aria-hidden="true"></i>
          <strong>${direction.name}</strong>
          <span>${direction.projects.length}</span>
        </a>
      `
    )
    .join("");
}

function projectMatches(project, activeFilter, query) {
  const filterMatch = activeFilter === "all" || project.direction.id === activeFilter;
  const text = [
    project.title,
    project.description,
    project.result,
    project.kind,
    project.direction.name,
    project.direction.english,
    project.year,
    ...project.tags
  ]
    .join(" ")
    .toLowerCase();
  return filterMatch && text.includes(query.toLowerCase());
}

function renderProjects() {
  const sections = document.querySelector("#projectSections");
  const emptyState = document.querySelector("#emptyState");
  const activeTab = document.querySelector(".direction-tab.is-active");
  const activeFilter = activeTab ? activeTab.dataset.filter : "all";
  const query = document.querySelector("#projectSearch")?.value.trim() || "";

  if (!sections) return;

  let visibleCount = 0;
  sections.innerHTML = directions
    .map((direction) => {
      const cards = direction.projects
        .filter((project) => projectMatches({ ...project, direction }, activeFilter, query))
        .map((project) => {
          visibleCount += 1;
          return `
            <article class="project-card" style="--accent: ${direction.accent}">
              <span class="project-kind">${project.kind}</span>
              <h3>${project.title}</h3>
              <p>${project.description}</p>
              <div class="tag-row">
                ${project.tags.map((tag) => `<span class="tag">${tag}</span>`).join("")}
              </div>
              <div class="result-line">${project.result}</div>
              <div class="link-row">
                ${project.links
                  .map(
                    (link) => `
                      <a class="demo-link" href="${link.url}" target="${link.url.startsWith("http") ? "_blank" : "_self"}" rel="noreferrer">
                        ${link.label}
                        <span aria-hidden="true">↗</span>
                      </a>
                    `
                  )
                  .join("")}
                ${
                  project.samples
                    ? `<a class="demo-link" href="demo.html?project=${project.id}">
                        本页试听
                        <span aria-hidden="true">↗</span>
                      </a>`
                    : ""
                }
              </div>
            </article>
          `;
        })
        .join("");

      if (!cards) return "";

      return `
        <section class="direction-section" id="${direction.id}" aria-labelledby="${direction.id}-title">
          <div class="direction-heading">
            <div>
              <span class="direction-badge" style="--accent: ${direction.accent}">${direction.english}</span>
              <h2 id="${direction.id}-title">${direction.name}</h2>
              <p>${direction.summary}</p>
            </div>
            <span class="direction-badge" style="--accent: ${direction.accent}">${direction.projects.length} 项</span>
          </div>
          <div class="project-list">
            ${cards}
          </div>
        </section>
      `;
    })
    .join("");

  if (emptyState) {
    emptyState.hidden = visibleCount > 0;
  }
}

function renderPublications() {
  const holder = document.querySelector("#publicationList");
  if (!holder) return;

  holder.innerHTML = publications
    .map(
      (item) => `
        <article>
          <span class="outcome-year">${item.year}</span>
          <h3>
            ${item.link
              ? `<a href="${item.link}" target="_blank" rel="noreferrer">${item.title}</a>`
              : item.title}
          </h3>
          <p>${item.authors}</p>
          <p>${item.venue}</p>
        </article>
      `
    )
    .join("");
}

function renderLabInfo() {
  const emailHolder = document.querySelector("#labEmails");
  if (emailHolder) {
    emailHolder.innerHTML = labInfo.emails
      .map((email) => `<a href="mailto:${email}">${email}</a>`)
      .join("");
  }

  const office = document.querySelector("#labOffice");
  if (office) office.textContent = labInfo.office;

  const profile = document.querySelector("#profileLink");
  if (profile) profile.href = labInfo.source;
}

function setupFilters() {
  const tabs = document.querySelectorAll(".direction-tab");
  const search = document.querySelector("#projectSearch");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((item) => item.classList.remove("is-active"));
      tab.classList.add("is-active");
      renderProjects();
    });
  });

  search?.addEventListener("input", renderProjects);
}

function drawSignalCanvas(canvas, phaseOffset = 0) {
  const ctx = canvas.getContext("2d");
  let width = 0;
  let height = 0;

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    width = Math.max(320, rect.width);
    height = Math.max(220, rect.height);
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function draw(time) {
    const t = time / 1000 + phaseOffset;
    ctx.clearRect(0, 0, width, height);

    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, "#111b18");
    grad.addColorStop(0.55, "#18221f");
    grad.addColorStop(1, "#252116");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.07)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= width; x += 38) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y <= height; y += 34) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const rows = 18;
    const cols = 44;
    const cellW = width / cols;
    const spectroTop = height * 0.12;
    const spectroH = height * 0.42;
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const pulse = Math.sin(col * 0.52 + t * 2.2) + Math.cos(row * 0.85 - t * 1.4);
        const ridge = Math.exp(-Math.abs(row - 7 - Math.sin(col * 0.25 + t) * 4) / 4);
        const energy = Math.max(0, Math.min(1, 0.18 + ridge * 0.7 + pulse * 0.08));
        const hueColor = row % 3 === 0 ? "8, 124, 112" : row % 3 === 1 ? "196, 85, 62" : "159, 113, 25";
        ctx.fillStyle = `rgba(${hueColor}, ${energy})`;
        ctx.fillRect(
          col * cellW + 2,
          spectroTop + row * (spectroH / rows) + 2,
          cellW - 4,
          spectroH / rows - 4
        );
      }
    }

    const center = height * 0.72;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(88, 208, 182, 0.94)";
    ctx.beginPath();
    for (let x = 0; x <= width; x += 3) {
      const amp = Math.sin(x * 0.032 + t * 2.4) * 34 + Math.sin(x * 0.011 - t * 1.3) * 19;
      const gate = 0.45 + Math.pow(Math.sin(x * 0.016 + t), 2) * 0.75;
      const y = center + amp * gate;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
    ctx.beginPath();
    for (let x = 0; x <= width; x += 5) {
      const y = height * 0.86 + Math.sin(x * 0.052 - t * 3.1) * 10;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = "rgba(255, 255, 255, 0.72)";
    ctx.font = "700 12px Inter, system-ui, sans-serif";
    ctx.fillText("XINJIANG UNIVERSITY · SPEECH PERCEPTION AND INTELLIGENT COMPUTING LABORATORY", 20, 30);


    window.requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", resize);
  window.requestAnimationFrame(draw);
}

function renderDemoSamples(project) {
  const holder = document.querySelector("#demoSamples");
  if (!holder) return;

  holder.innerHTML = project.samples
    .map(
      (sample) => `
        <article class="sample-card">
          <h2>${sample.title}</h2>
          <p>${sample.text}</p>
          <div class="sample-table" role="table" aria-label="${sample.title} 音频示例">
            <div class="sample-table-head" role="row">
              <span role="columnheader">Method</span>
              <span role="columnheader">Audio</span>
            </div>
            ${sample.tracks
              .map(
                (track) => `
                  <div class="sample-table-row" role="row">
                    <strong role="cell">${track.label}</strong>
                    <div role="cell">
                      <audio controls preload="none" src="${track.src}"></audio>
                    </div>
                  </div>
                `
              )
              .join("")}
          </div>
        </article>
      `
    )
    .join("");
}

function setupDemoPage() {
  const demoCanvas = document.querySelector("#demoCanvas");
  if (!demoCanvas) return;

  const params = new URLSearchParams(window.location.search);
  const id = params.get("project") || "dytswig";
  const project = projects.find((item) => item.id === id) || projects[0];

  document.title = `${project.title} | B415 Demo`;
  document.querySelector("#demoDirection").textContent = project.direction.name;
  document.querySelector("#demoTitle").textContent = project.title;
  document.querySelector("#demoDescription").textContent = project.description;
  document.querySelector("#demoTags").innerHTML = project.tags
    .map((tag) => `<span class="tag" style="--accent: ${project.direction.accent}">${tag}</span>`)
    .join("");
  document.querySelector("#demoLinks").innerHTML = project.links
    .map(
      (link) => `
        <a class="demo-link" href="${link.url}" target="${link.url.startsWith("http") ? "_blank" : "_self"}" rel="noreferrer">
          ${link.label}
          <span aria-hidden="true">↗</span>
        </a>
      `
    )
    .join("");

  renderDemoSamples(project);
  drawSignalCanvas(demoCanvas, 2.8);
}

function init() {
  renderLabInfo();
  renderDirectionIndex();
  renderProjects();
  renderPublications();
  setupFilters();

  const signalCanvas = document.querySelector("#signalCanvas");
  if (signalCanvas) {
    drawSignalCanvas(signalCanvas);
  }

  setupDemoPage();
}

init();
