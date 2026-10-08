import json
import pubchempy as pcp
import PyPDF2
from google import genai
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# إعدادات صفحة الويب
# ==========================================
st.set_page_config(page_title="MedChem AI Studio", page_icon="💊", layout="wide")
st.title("💊 تطبيق الكيمياء الصيدلية الذكي (MedChem AI)")

# الشريط الجانبي (Sidebar)
st.sidebar.header("⚙️ الإعدادات والتحكم")
API_KEY = st.sidebar.text_input("أدخل مفتاح Gemini API الخاص بك:", type="password")


# ==========================================
# الدوال الأساسية (Backend)
# ==========================================
def extract_text_from_pdf(uploaded_file):
  text = ""
  reader = PyPDF2.PdfReader(uploaded_file)
  for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
      text += extracted + "\n"
  return text


def analyze_lecture_with_ai(text, client):
  system_instruction = (
      "You are an expert medicinal chemistry tutor. Analyze the text and extract"
      " drugs. Return STRICTLY a valid JSON object. Keys are drug names. Values"
      " are objects with: 'summary': (1 short sentence mechanism/use), "
      "'sar_notes': (1 sentence analyzing its SAR), 'pharmacophore_smarts': (A"
      " valid RDKit SMARTS string of its key pharmacophore to highlight. or"
      " empty string)."
  )
  prompt_text = text[:15000]

  response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=f"Extract drugs and their SAR data:\n{prompt_text}",
      config={"system_instruction": system_instruction},
  )

  clean_json = response.text.strip().replace("```json", "").replace("```", "")
  try:
    return json.loads(clean_json)
  except json.JSONDecodeError:
    return {}


def get_smiles_from_pubchem(compound_name):
  try:
    compounds = pcp.get_compounds(compound_name, "name")
    if compounds:
      return compounds[0].isomeric_smiles
  except Exception:
    pass
  return None


def generate_svg(mol, smarts_pattern=""):
  rdDepictor.SetPreferCoordGen(True)
  rdDepictor.Compute2DCoords(mol)
  drawer = rdMolDraw2D.MolDraw2DSVG(350, 300)

  highlight_atoms = []
  if smarts_pattern:
    patt = Chem.MolFromSmarts(smarts_pattern)
    if patt and mol.HasSubstructMatch(patt):
      highlight_atoms = list(mol.GetSubstructMatch(patt))

  opts = drawer.drawOptions()
  opts.setHighlightColour((1.0, 0.4, 0.4, 0.5))
  drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
  drawer.FinishDrawing()
  return drawer.GetDrawingText().replace("svg:", "").replace("xmlns:svg", "xmlns")


def get_interactive_board_html(drugs_data, hide_details=False):
  """توليد اللوحة التفاعلية مع دعم وضع الاختبار الذاتي Flashcards"""
  cards_html = ""
  for index, (name, data) in enumerate(drugs_data.items()):
    left_pos = (index % 4) * 380 + 20
    top_pos = (index // 4) * 480 + 20

    if hide_details:
      # وضع الاختبار: إخفاء الاسم والـ SAR واستبداله بزر كشف البيانات
      display_title = f"مركب رقم {index + 1}"
      card_content = f"""
            <h3 class="masked-title" id="title-{index}">{display_title}</h3>
            <button class="reveal-btn" onclick="revealCard({index}, '{name}')">👁️ كشف اسم الدواء والـ SAR</button>
            <div class="drug-svg">{data.get('svg', '')}</div>
            <div class="sar-box masked-box" id="sar-{index}" style="display:none;">
                <p><strong>{name}</strong></p>
                <p>{data.get('summary', '')}</p>
                <hr>
                <strong>💡 SAR:</strong> {data.get('sar_notes', '')}
            </div>
            """
    else:
      # الوضع العادي
      card_content = f"""
            <h3>{name}</h3>
            <p class="summary">{data.get('summary', '')}</p>
            <div class="drug-svg">{data.get('svg', '')}</div>
            <div class="sar-box"><strong>💡 SAR:</strong> {data.get('sar_notes', '')}</div>
            """

    cards_html += f"""
        <div class="card" style="left: {left_pos}px; top: {top_pos}px;">
            {card_content}
        </div>
        """

  return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/@panzoom/panzoom@4.5.1/dist/panzoom.min.js"></script>
        <style>
            body {{ margin: 0; background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; overflow: hidden; }}
            #board-container {{ width: 100vw; height: 100vh; cursor: grab; }}
            #board {{ width: 5000px; height: 5000px; position: relative; }}
            .card {{
                position: absolute; background: white; border-radius: 12px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1); padding: 15px;
                cursor: grab; text-align: center; width: 330px; border-top: 4px solid #3498db;
            }}
            h3 {{ margin: 0 0 8px 0; color: #2c3e50; }}
            .summary {{ font-size: 13px; color: #7f8c8d; font-style: italic; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            .drug-svg svg {{ width: 100%; height: auto; }}
            .sar-box {{ background: #e8f4f8; color: #1f618d; padding: 10px; border-radius: 8px; font-size: 13px; text-align: left; border-left: 4px solid #3498db; margin-top: 10px;}}
            .reveal-btn {{ background: #e74c3c; color: white; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; margin-bottom: 8px; }}
            .reveal-btn:hover {{ background: #c0392b; }}
        </style>
    </head>
    <body>
        <div id="board-container"><div id="board">{cards_html}</div></div>
        <script>
            const container = document.getElementById('board-container');
            const board = document.getElementById('board');
            const panzoom = Panzoom(board, {{ maxScale: 3, minScale: 0.2 }});
            container.addEventListener('wheel', panzoom.zoomWithWheel);
            
            document.querySelectorAll('.card').forEach(card => {{
                let isDragging = false, startX, startY, initialLeft, initialTop;
                card.addEventListener('mousedown', (e) => {{
                    isDragging = true; e.stopPropagation();
                    startX = e.clientX; startY = e.clientY;
                    initialLeft = parseFloat(card.style.left) || 0; initialTop = parseFloat(card.style.top) || 0;
                }});
                window.addEventListener('mousemove', (e) => {{
                    if (!isDragging) return;
                    const scale = panzoom.getScale();
                    card.style.left = `${{initialLeft + (e.clientX - startX) / scale}}px`; 
                    card.style.top = `${{initialTop + (e.clientY - startY) / scale}}px`;
                }});
                window.addEventListener('mouseup', () => isDragging = false);
            }});

            function revealCard(index, fullName) {{
                document.getElementById('title-' + index).innerText = fullName;
                document.getElementById('sar-' + index).style.display = 'block';
            }}
        </script>
    </body>
    </html>
    """


# ==========================================
# واجهة المستخدم (UI)
# ==========================================
tab1, tab2 = st.tabs([
    "📄 صانع خرائط المحاضرات (PDF Board)",
    "🧪 التعديل الدوائي الذكي (AI SAR)",
])

# ----- التبويب الأول -----
with tab1:
  st.header("صناعة لوحة مراجعة تفاعلية من ملزمة المحاضرة")
  uploaded_file = st.file_uploader("ارفع ملف الـ PDF هنا:", type="pdf")

  if st.button("تحليل وبناء الخريطة 🚀"):
    if not API_KEY:
      st.error("الرجاء إدخال مفتاح الـ API في الشريط الجانبي أولاً.")
    elif uploaded_file is not None:
      client = genai.Client(api_key=API_KEY.strip())
      with st.spinner("جاري قراءة الملف وتلخيص الـ SAR وتحديد الـ Pharmacophores..."):
        text = extract_text_from_pdf(uploaded_file)
        st.session_state["lecture_text"] = text
        drugs_dict = analyze_lecture_with_ai(text, client)

        if drugs_dict:
          drugs_data = {}
          for name, data in drugs_dict.items():
            smiles = get_smiles_from_pubchem(name)
            if smiles:
              mol = Chem.MolFromSmiles(smiles)
              if mol:
                data["svg"] = generate_svg(mol, data.get("pharmacophore_smarts", ""))
                drugs_data[name] = data

          if drugs_data:
            st.session_state["drugs_data"] = drugs_data
            st.success(f"تم بنجاح بناء الخريطة الذهنية لـ ({len(drugs_data)}) مركب!")

  if "drugs_data" in st.session_state:
    drugs_list = list(st.session_state["drugs_data"].keys())

    # خيارات التحكم والمقارنة في الشريط الجانبي
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 ميزات المذاكرة المتقدمة")

    # 1. مفتاح وضع الاختبار
    test_mode = st.sidebar.toggle("🧠 وضع البطاقات الومضية (اختبر نفسك)")

    # 2. أداة المقارنة المباشرة
    st.sidebar.markdown("**⚖️ مقارنة مركبين:**")
    drug_a = st.sidebar.selectbox("المركب الأول:", drugs_list, index=0)
    drug_b = st.sidebar.selectbox(
        "المركب الثاني:", drugs_list, index=min(1, len(drugs_list) - 1)
    )

    if st.sidebar.button("إنشاء جدول مقارنة 📊"):
      if API_KEY:
        client = genai.Client(api_key=API_KEY.strip())
        prompt = f"Compare in detail between {drug_a} and {drug_b} based on structural differences, mechanism, and SAR. Provide the response as a clear markdown comparison table followed by bullet points."
        with st.spinner("جاري إعداد جدول المقارنة..."):
          res = client.models.generate_content(
              model="gemini-2.5-flash", contents=prompt
          )
          if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
          st.session_state["chat_history"].append(
              {"role": "user", "content": f"قارن بين {drug_a} و {drug_b}"}
          )
          st.session_state["chat_history"].append(
              {"role": "assistant", "content": res.text}
          )

    # 3. زر تصدير الخريطة
    html_board_code = get_interactive_board_html(
        st.session_state["drugs_data"], hide_details=test_mode
    )
    st.sidebar.download_button(
        label="💾 تصدير الخريطة كملف HTML",
        data=html_board_code,
        file_name="MedChem_MindMap.html",
        mime="text/html",
    )

    # عرض اللوحة والشات
    col_board, col_chat = st.columns([3, 1])

    with col_board:
      components.html(html_board_code, height=750, scrolling=False)

    with col_chat:
      st.subheader("💬 المعلم الصيدلي الذكي")
      if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

      for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
          st.write(msg["content"])

      user_question = st.chat_input("اسأل عن أي مركب أو مقارنة...")
      if user_question:
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
          st.write(user_question)

        if API_KEY:
          client = genai.Client(api_key=API_KEY.strip())
          lecture_context = st.session_state.get("lecture_text", "")[:10000]
          system_instruction = (
              "You are an expert medicinal chemistry tutor. Answer based on"
              " lecture context and SAR knowledge."
          )
          prompt = (
              f"Lecture Context:\n{lecture_context}\n\nUser Question:"
              f" {user_question}"
          )

          with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
              response = client.models.generate_content(
                  model="gemini-2.5-flash",
                  contents=prompt,
                  config={"system_instruction": system_instruction},
              )
              st.write(response.text)
              st.session_state["chat_history"].append(
                  {"role": "assistant", "content": response.text}
              )

# ----- التبويب الثاني -----
with tab2:
  st.header("التعديل الحر على الأدوية (AI Modification)")
  user_prompt = st.text_area(
      "أدخل طلبك (مثال: ارسم الباراسيتامول واستبدل مجموعة الهيدروكسيل بـ فلور"
      " واشرح التأثير):"
  )

  if st.button("تعديل وتحليل 🧪"):
    if not API_KEY:
      st.error("الرجاء إدخال مفتاح الـ API في الشريط الجانبي.")
    elif user_prompt:
      client = genai.Client(api_key=API_KEY.strip())
      with st.spinner("الذكاء الاصطناعي يقوم بحساب التعديلات..."):
        system_instruction = (
            "You are a medicinal chemist. Return STRICTLY a JSON object:"
            " {'smiles': 'valid smiles of NEW compound', 'explanation': 'SAR"
            " impact analysis'}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config={"system_instruction": system_instruction},
        )
        clean_json = (
            response.text.strip().replace("```json", "").replace("```", "")
        )

        try:
          result = json.loads(clean_json)
          mol = Chem.MolFromSmiles(result.get("smiles", ""))
          if mol:
            svg_image = generate_svg(mol)
            col1, col2 = st.columns([1, 2])
            with col1:
              st.components.v1.html(svg_image, width=350, height=300)
            with col2:
              st.info("💡 **تحليل المعلم الصيدلي (SAR):**")
              st.write(result.get("explanation", ""))
          else:
            st.error("الصيغة الكيميائية الناتجة غير صحيحة.")
        except:
          st.error("حدث خطأ في فهم طلبك، حاول بصياغة مختلفة.")