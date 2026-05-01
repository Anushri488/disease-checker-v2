from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import json
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)
CORS(app)

model     = joblib.load("model.pkl")
le        = joblib.load("encoder.pkl")
all_symptoms = joblib.load("symptoms_list.pkl")

# ─────────────────────────────────────────────────
# DISEASE METADATA  (description + severity + doctor warning)
# ─────────────────────────────────────────────────
DISEASE_INFO = {
    "Fungal infection": {
        "description": "A common infection caused by fungi affecting skin, nails, or mucous membranes. Usually superficial but can spread without treatment.",
        "severity": "mild",
        "doctor_advice": "Visit a doctor if rash spreads, doesn't improve in 2 weeks, or affects nails/scalp.",
        "emergency": False,
        "precautions": ["Keep affected area clean and dry", "Avoid sharing towels or clothing", "Use antifungal powder/cream"]
    },
    "Allergy": {
        "description": "An immune system overreaction to substances like pollen, dust, or food. Symptoms range from mild sneezing to severe anaphylaxis.",
        "severity": "mild",
        "doctor_advice": "See a doctor if symptoms interfere with daily life or you suspect food/drug allergy.",
        "emergency": False,
        "precautions": ["Identify and avoid allergens", "Keep antihistamines handy", "Wear a mask outdoors during pollen season"]
    },
    "GERD": {
        "description": "Gastroesophageal reflux disease — stomach acid frequently flows back into the esophagus, causing heartburn and irritation.",
        "severity": "moderate",
        "doctor_advice": "See a doctor if symptoms occur more than twice a week or don't respond to antacids.",
        "emergency": False,
        "precautions": ["Avoid spicy, fatty or acidic foods", "Don't lie down right after eating", "Elevate head of bed"]
    },
    "Chronic cholestasis": {
        "description": "Reduced bile flow from the liver, causing bile to build up in the bloodstream. Can lead to liver damage over time.",
        "severity": "severe",
        "doctor_advice": "See a doctor immediately — this condition requires medical evaluation and liver function tests.",
        "emergency": False,
        "precautions": ["Avoid alcohol completely", "Follow a low-fat diet", "Take prescribed liver medications"]
    },
    "Drug Reaction": {
        "description": "An adverse response to a medication, ranging from mild rashes to life-threatening reactions like Stevens-Johnson syndrome.",
        "severity": "moderate",
        "doctor_advice": "Stop the suspected medication and see a doctor immediately if you have skin blistering or difficulty breathing.",
        "emergency": True,
        "precautions": ["Stop the suspected medication", "Document the drug name", "Carry a list of drug allergies"]
    },
    "Peptic ulcer disease": {
        "description": "Open sores in the lining of the stomach, small intestine, or esophagus, often caused by H. pylori bacteria or NSAIDs.",
        "severity": "moderate",
        "doctor_advice": "See a doctor if pain is severe, you notice black stools, or vomit blood.",
        "emergency": False,
        "precautions": ["Avoid NSAIDs like ibuprofen", "Quit smoking", "Limit alcohol and spicy food"]
    },
    "AIDS": {
        "description": "Advanced stage of HIV infection where the immune system is severely damaged. Requires lifelong antiretroviral therapy.",
        "severity": "severe",
        "doctor_advice": "Immediate medical care required. Regular CD4 count and viral load monitoring is essential.",
        "emergency": True,
        "precautions": ["Start antiretroviral therapy immediately", "Practice safe sex", "Avoid sharing needles"]
    },
    "Diabetes": {
        "description": "A metabolic disease causing high blood sugar due to insufficient insulin production or action. Type 2 is the most common form.",
        "severity": "moderate",
        "doctor_advice": "See a doctor for blood glucose testing. Uncontrolled diabetes leads to serious organ damage.",
        "emergency": False,
        "precautions": ["Monitor blood sugar regularly", "Follow a low-glycemic diet", "Exercise at least 30 min/day"]
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines, usually caused by viral or bacterial infection. Often called 'stomach flu'.",
        "severity": "mild",
        "doctor_advice": "See a doctor if you cannot keep fluids down, have bloody diarrhea, or symptoms last more than 3 days.",
        "emergency": False,
        "precautions": ["Stay hydrated with ORS solution", "Eat bland foods (BRAT diet)", "Wash hands frequently"]
    },
    "Bronchial Asthma": {
        "description": "A chronic condition causing airways to narrow and swell, producing extra mucus. Triggers include allergens, exercise, and cold air.",
        "severity": "moderate",
        "doctor_advice": "See a doctor if you use your rescue inhaler more than twice a week. Severe attacks need emergency care.",
        "emergency": False,
        "precautions": ["Carry a rescue inhaler always", "Identify and avoid triggers", "Get flu vaccination annually"]
    },
    "Hypertension": {
        "description": "Persistently elevated blood pressure (≥140/90 mmHg) that strains the heart and blood vessels, often called the 'silent killer'.",
        "severity": "moderate",
        "doctor_advice": "See a doctor to get blood pressure checked. Go to ER if BP reading is above 180/120.",
        "emergency": False,
        "precautions": ["Reduce salt intake", "Exercise regularly", "Monitor BP at home daily"]
    },
    "Migraine": {
        "description": "A neurological condition causing intense, throbbing headaches often accompanied by nausea, vomiting, and light sensitivity.",
        "severity": "moderate",
        "doctor_advice": "See a doctor if migraines are frequent (>4/month), very severe, or accompanied by neurological symptoms.",
        "emergency": False,
        "precautions": ["Identify and avoid migraine triggers", "Rest in a quiet dark room during attacks", "Stay hydrated"]
    },
    "Cervical spondylosis": {
        "description": "Age-related wear and tear of the neck vertebrae causing neck pain, stiffness, and sometimes nerve compression.",
        "severity": "mild",
        "doctor_advice": "See a doctor if you experience numbness in arms/hands or loss of bladder/bowel control.",
        "emergency": False,
        "precautions": ["Use an ergonomic pillow", "Practice neck exercises", "Avoid prolonged screen time"]
    },
    "Jaundice": {
        "description": "Yellowing of the skin and eyes caused by excess bilirubin in the blood, often indicating liver or bile duct problems.",
        "severity": "severe",
        "doctor_advice": "See a doctor immediately — jaundice always requires medical evaluation to find the underlying cause.",
        "emergency": True,
        "precautions": ["Avoid alcohol completely", "Rest adequately", "Follow prescribed diet"]
    },
    "Malaria": {
        "description": "A life-threatening parasitic disease transmitted by infected mosquitoes. Common in tropical regions, causing cyclical high fevers.",
        "severity": "severe",
        "doctor_advice": "Go to a doctor or hospital immediately for a blood smear test. Malaria can be fatal if untreated.",
        "emergency": True,
        "precautions": ["Take prescribed antimalarial medication", "Use mosquito nets and repellents", "Wear full-sleeve clothing at dusk"]
    },
    "Chicken pox": {
        "description": "A highly contagious viral infection causing an itchy rash with blisters. Usually mild in children but can be serious in adults.",
        "severity": "mild",
        "doctor_advice": "See a doctor if you're pregnant, immunocompromised, or if rash spreads to eyes. Adults need medical supervision.",
        "emergency": False,
        "precautions": ["Stay isolated to prevent spread", "Avoid scratching the blisters", "Use calamine lotion for itching"]
    },
    "Dengue": {
        "description": "A mosquito-borne viral infection causing severe flu-like illness. Severe dengue can cause plasma leakage, organ failure, and death.",
        "severity": "severe",
        "doctor_advice": "See a doctor immediately for a platelet count test. Watch for warning signs: severe abdominal pain, bleeding gums.",
        "emergency": True,
        "precautions": ["Get platelet count monitored daily", "Stay well hydrated", "Eliminate mosquito breeding sites"]
    },
    "Typhoid": {
        "description": "A bacterial infection caused by Salmonella typhi, spread through contaminated food and water. Causes sustained high fever.",
        "severity": "severe",
        "doctor_advice": "See a doctor for a Widal test and blood culture. Antibiotics are required — don't self-medicate.",
        "emergency": True,
        "precautions": ["Take complete antibiotic course", "Drink only boiled or bottled water", "Eat freshly cooked food only"]
    },
    "Hepatitis A": {
        "description": "A liver infection caused by the Hepatitis A virus, spread through contaminated food and water. Usually self-limiting.",
        "severity": "moderate",
        "doctor_advice": "See a doctor for liver function tests. Most cases resolve on their own but monitoring is important.",
        "emergency": False,
        "precautions": ["Rest adequately", "Avoid alcohol", "Get Hepatitis A vaccination for prevention"]
    },
    "Hepatitis B": {
        "description": "A viral liver infection that can become chronic, leading to cirrhosis or liver cancer. Spread through blood and body fluids.",
        "severity": "severe",
        "doctor_advice": "See a doctor for HBsAg test and liver function evaluation. Chronic cases need antiviral treatment.",
        "emergency": False,
        "precautions": ["Get vaccinated (3-dose series)", "Avoid sharing needles or razors", "Practice safe sex"]
    },
    "Hepatitis C": {
        "description": "A viral infection causing liver inflammation, often leading to chronic liver disease. Spread primarily through blood contact.",
        "severity": "severe",
        "doctor_advice": "See a doctor for HCV antibody test. Modern antiviral treatments can cure Hepatitis C in most cases.",
        "emergency": False,
        "precautions": ["Avoid sharing needles", "Don't share personal care items", "Get tested if you had blood transfusions before 1992"]
    },
    "Hepatitis D": {
        "description": "A liver infection that only occurs in people already infected with Hepatitis B. Causes more severe disease.",
        "severity": "severe",
        "doctor_advice": "Immediate medical care required. Combined HBV/HDV infection has worse prognosis.",
        "emergency": True,
        "precautions": ["Hepatitis B vaccination prevents HDV too", "Avoid contact with infected blood", "Seek specialist liver care"]
    },
    "Hepatitis E": {
        "description": "A liver disease caused by Hepatitis E virus, mainly spread through contaminated water. Can be fatal in pregnant women.",
        "severity": "severe",
        "doctor_advice": "See a doctor immediately, especially if pregnant. Can cause acute liver failure.",
        "emergency": True,
        "precautions": ["Drink only safe, treated water", "Maintain good hand hygiene", "Avoid raw or undercooked shellfish"]
    },
    "Alcoholic hepatitis": {
        "description": "Liver inflammation caused by drinking too much alcohol over time. Can rapidly progress to liver failure.",
        "severity": "severe",
        "doctor_advice": "Immediate medical attention required. Complete alcohol cessation is essential for survival.",
        "emergency": True,
        "precautions": ["Stop drinking alcohol completely", "Seek alcohol addiction counseling", "Follow prescribed nutritional plan"]
    },
    "Tuberculosis": {
        "description": "A serious bacterial infection primarily affecting the lungs, spread through the air when infected people cough or sneeze.",
        "severity": "severe",
        "doctor_advice": "See a doctor immediately for sputum test and chest X-ray. TB is curable but requires 6 months of treatment.",
        "emergency": True,
        "precautions": ["Take all prescribed medications for full 6 months", "Cover mouth when coughing", "Ensure good ventilation at home"]
    },
    "Common Cold": {
        "description": "A viral infection of the upper respiratory tract, usually harmless. Caused by rhinoviruses most commonly.",
        "severity": "mild",
        "doctor_advice": "See a doctor if fever exceeds 39°C, symptoms worsen after 10 days, or you have difficulty breathing.",
        "emergency": False,
        "precautions": ["Rest and stay hydrated", "Wash hands frequently", "Avoid close contact with others"]
    },
    "Pneumonia": {
        "description": "Infection that inflames air sacs in one or both lungs, which may fill with fluid. Can be life-threatening in elderly.",
        "severity": "severe",
        "doctor_advice": "See a doctor or go to ER immediately. Pneumonia can rapidly worsen, especially in elderly or immunocompromised.",
        "emergency": True,
        "precautions": ["Get pneumococcal and flu vaccines", "Complete antibiotic course", "Rest completely and stay hydrated"]
    },
    "Heart attack": {
        "description": "A medical emergency where blood flow to part of the heart is blocked. Every minute without treatment causes more heart muscle death.",
        "severity": "severe",
        "doctor_advice": "CALL EMERGENCY SERVICES IMMEDIATELY (102/112). Do not drive yourself. Chew aspirin if not allergic.",
        "emergency": True,
        "precautions": ["Call emergency services immediately", "Chew 325mg aspirin if available and not allergic", "Do not eat or drink anything"]
    },
    "Varicose veins": {
        "description": "Enlarged, twisted veins visible just under the skin surface, usually in legs. Caused by weak or damaged vein valves.",
        "severity": "mild",
        "doctor_advice": "See a doctor if you have significant pain, swelling, or skin changes. Seek emergency care for sudden severe pain or bleeding.",
        "emergency": False,
        "precautions": ["Elevate legs when resting", "Wear compression stockings", "Exercise regularly to improve circulation"]
    },
    "Hypothyroidism": {
        "description": "Underactive thyroid gland that doesn't produce enough thyroid hormones, slowing metabolism and affecting all body systems.",
        "severity": "moderate",
        "doctor_advice": "See a doctor for thyroid function tests (TSH, T3, T4). Requires lifelong thyroid hormone replacement.",
        "emergency": False,
        "precautions": ["Take thyroid medication at the same time daily", "Avoid raw cruciferous vegetables in excess", "Get TSH checked every 6-12 months"]
    },
    "Hyperthyroidism": {
        "description": "Overactive thyroid producing excess thyroid hormones, accelerating body's metabolism causing weight loss, rapid heartbeat.",
        "severity": "moderate",
        "doctor_advice": "See a doctor for thyroid tests. Thyroid storm (extreme hyperthyroidism) is a medical emergency.",
        "emergency": False,
        "precautions": ["Take anti-thyroid medications as prescribed", "Avoid excess iodine", "Monitor heart rate daily"]
    },
    "Hypoglycemia": {
        "description": "Abnormally low blood sugar levels (< 70 mg/dL), causing symptoms from shakiness to unconsciousness if severe.",
        "severity": "moderate",
        "doctor_advice": "If unconscious or unable to swallow, call emergency services immediately. Recurring episodes need doctor evaluation.",
        "emergency": False,
        "precautions": ["Always carry fast-acting sugar (glucose tablets)", "Eat regular meals — don't skip", "Monitor blood sugar regularly"]
    },
    "Osteoarthritis": {
        "description": "A degenerative joint disease where cartilage breaks down, causing pain, stiffness, and reduced mobility. Most common in knees, hips.",
        "severity": "moderate",
        "doctor_advice": "See a doctor for X-rays and pain management plan. Severe cases may need joint replacement surgery.",
        "emergency": False,
        "precautions": ["Maintain healthy weight to reduce joint stress", "Do low-impact exercise like swimming", "Use hot/cold therapy for pain relief"]
    },
    "Arthritis": {
        "description": "Inflammation of one or more joints causing pain and stiffness. Rheumatoid arthritis is an autoimmune condition.",
        "severity": "moderate",
        "doctor_advice": "See a doctor for blood tests (RF, anti-CCP). Early treatment prevents joint damage in rheumatoid arthritis.",
        "emergency": False,
        "precautions": ["Take prescribed anti-inflammatory medications", "Do gentle range-of-motion exercises", "Use assistive devices to reduce joint stress"]
    },
    "Vertigo": {
        "description": "A sensation of dizziness where surroundings appear to spin. Often caused by inner ear problems (BPPV) or central nervous system issues.",
        "severity": "mild",
        "doctor_advice": "See a doctor if vertigo is sudden and severe, accompanied by hearing loss, or you have neurological symptoms.",
        "emergency": False,
        "precautions": ["Move slowly, especially when getting up", "Avoid sudden head movements", "Do Epley maneuver exercises (Physio guidance)"]
    },
    "Acne": {
        "description": "A skin condition where hair follicles become clogged with oil and dead skin cells, causing pimples, blackheads, and cysts.",
        "severity": "mild",
        "doctor_advice": "See a dermatologist for severe or cystic acne, or if OTC treatments fail after 2-3 months.",
        "emergency": False,
        "precautions": ["Cleanse face twice daily with gentle cleanser", "Don't squeeze pimples", "Use non-comedogenic skincare products"]
    },
    "Urinary tract infection": {
        "description": "A bacterial infection in any part of the urinary system. More common in women. Can spread to kidneys if untreated.",
        "severity": "moderate",
        "doctor_advice": "See a doctor for urine culture. If you have fever, back pain, or chills, the infection may have reached kidneys.",
        "emergency": False,
        "precautions": ["Drink plenty of water", "Urinate after sexual intercourse", "Avoid holding urine for long periods"]
    },
    "Psoriasis": {
        "description": "A chronic autoimmune condition causing rapid buildup of skin cells, resulting in scaling, inflammation, and red patches.",
        "severity": "mild",
        "doctor_advice": "See a dermatologist. Psoriatic arthritis is a serious complication requiring rheumatology care.",
        "emergency": False,
        "precautions": ["Moisturize skin regularly", "Avoid smoking and limit alcohol", "Manage stress which triggers flares"]
    },
    "Impetigo": {
        "description": "A highly contagious bacterial skin infection causing red sores that rupture, ooze, and form yellow-brown crusts.",
        "severity": "mild",
        "doctor_advice": "See a doctor for antibiotic prescription. Keep child home from school until 24-48 hrs after starting treatment.",
        "emergency": False,
        "precautions": ["Keep sores covered", "Wash hands frequently", "Don't share towels or clothing"]
    },
    "Dimorphic hemmorhoids": {
        "description": "Swollen and inflamed veins in the rectum and anus (hemorrhoids/piles) causing discomfort, bleeding, and pain during bowel movements.",
        "severity": "mild",
        "doctor_advice": "See a doctor if rectal bleeding is significant, or piles are protruding and cannot be pushed back.",
        "emergency": False,
        "precautions": ["Eat a high-fiber diet", "Drink 8+ glasses of water daily", "Avoid straining during bowel movements"]
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Sudden loss of muscle function due to bleeding in or around the brain. A neurosurgical emergency requiring immediate intervention.",
        "severity": "severe",
        "doctor_advice": "CALL EMERGENCY SERVICES IMMEDIATELY (102/112). Time is critical — brain cells die every minute.",
        "emergency": True,
        "precautions": ["Call emergency services immediately", "Keep patient still and comfortable", "Do not give food or water"]
    },
    "Cervical Cancer": {
        "description": "Cancer of the cervix, usually caused by HPV infection. Highly preventable and treatable when detected early through screening.",
        "severity": "severe",
        "doctor_advice": "See a gynecologist immediately for colposcopy and biopsy. Regular Pap smears are critical for early detection.",
        "emergency": True,
        "precautions": ["Get HPV vaccination", "Get regular Pap smear tests", "Practice safe sex"]
    },
}

# Symptom severity classification
SYMPTOM_SEVERITY = {
    # Severe symptoms
    "breathlessness": "severe", "chest_pain": "severe", "blood_in_sputum": "severe",
    "high_fever": "severe", "altered_sensorium": "severe", "slurred_speech": "severe",
    "fast_heart_rate": "severe", "acute_liver_failure": "severe", "coma": "severe",
    "bloody_stool": "severe", "palpitations": "severe", "fluid_overload": "severe",
    "dehydration": "severe", "yellowing_of_eyes": "severe", "yellowish_skin": "severe",

    # Moderate symptoms
    "fatigue": "moderate", "vomiting": "moderate", "diarrhoea": "moderate",
    "fever": "moderate", "nausea": "moderate", "headache": "moderate",
    "abdominal_pain": "moderate", "joint_pain": "moderate", "muscle_pain": "moderate",
    "dark_urine": "moderate", "weight_loss": "moderate", "sweating": "moderate",
    "dizziness": "moderate", "loss_of_balance": "moderate", "blurred_and_distorted_vision": "moderate",
    "depression": "moderate", "anxiety": "moderate", "swelling_joints": "moderate",
    "burning_micturition": "moderate", "mild_fever": "moderate",

    # Mild symptoms (everything else defaults to mild)
}

def get_symptom_severity(symptom):
    return SYMPTOM_SEVERITY.get(symptom, "mild")


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    enriched = [
        {"name": s, "severity": get_symptom_severity(s)}
        for s in all_symptoms
    ]
    return jsonify({"symptoms": enriched, "symptom_names": all_symptoms})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_symptoms = set(s.strip() for s in data.get("symptoms", []))
    if not user_symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    input_vec = np.array(
        [1 if s in user_symptoms else 0 for s in all_symptoms]
    ).reshape(1, -1)

    pred_idx = model.predict(input_vec)[0]
    disease = le.inverse_transform([pred_idx])[0]

    proba = model.predict_proba(input_vec)[0]
    confidence = round(float(proba[pred_idx]) * 100, 1)

    top3 = []
    top_indices = np.argsort(proba)[::-1][:3]
    for idx in top_indices:
        top3.append({
            "disease": le.inverse_transform([idx])[0],
            "confidence": round(float(proba[idx]) * 100, 1)
        })

    matched = list(user_symptoms & set(all_symptoms))
    info = DISEASE_INFO.get(disease, {
        "description": "No additional information available.",
        "severity": "moderate",
        "doctor_advice": "Consult a doctor for proper diagnosis.",
        "emergency": False,
        "precautions": []
    })

    return jsonify({
        "disease": disease,
        "confidence": confidence,
        "top3": top3,
        "matched_symptoms": matched,
        "description": info["description"],
        "severity": info["severity"],
        "doctor_advice": info["doctor_advice"],
        "emergency": info["emergency"],
        "precautions": info.get("precautions", []),
        "symptom_severities": {s: get_symptom_severity(s) for s in matched},
        "timestamp": datetime.now().isoformat()
    })


@app.route("/pdf-report", methods=["POST"])
def pdf_report():
    data = request.get_json()
    disease     = data.get("disease", "Unknown")
    confidence  = data.get("confidence", 0)
    symptoms    = data.get("symptoms", [])
    top3        = data.get("top3", [])
    description = data.get("description", "")
    severity    = data.get("severity", "moderate")
    doctor_advice = data.get("doctor_advice", "")
    emergency   = data.get("emergency", False)
    precautions = data.get("precautions", [])
    sym_severities = data.get("symptom_severities", {})
    timestamp   = data.get("timestamp", datetime.now().isoformat())

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle("title",  fontSize=22, textColor=colors.HexColor("#0f172a"),
                                  fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6)
    sub_style    = ParagraphStyle("sub",    fontSize=10, textColor=colors.HexColor("#64748b"),
                                  fontName="Helvetica", alignment=TA_CENTER, spaceAfter=20)
    h2_style     = ParagraphStyle("h2",     fontSize=13, textColor=colors.HexColor("#0f172a"),
                                  fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6)
    body_style   = ParagraphStyle("body",   fontSize=10, textColor=colors.HexColor("#374151"),
                                  fontName="Helvetica", leading=15, spaceAfter=6)
    warn_style   = ParagraphStyle("warn",   fontSize=10, textColor=colors.HexColor("#92400e"),
                                  fontName="Helvetica-Bold", leading=14)
    emerg_style  = ParagraphStyle("emerg",  fontSize=11, textColor=colors.HexColor("#991b1b"),
                                  fontName="Helvetica-Bold", leading=14)

    SEVERITY_COLOR_MAP = {"mild": "#10b981", "moderate": "#f59e0b", "severe": "#ef4444"}
    sev_color = SEVERITY_COLOR_MAP.get(severity, "#f59e0b")

    story = []
    # Header
    story.append(Paragraph("🩺 SymptomAI — Disease Report", title_style))
    story.append(Paragraph(f"Generated on {timestamp[:10]} at {timestamp[11:19]}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 12))

    # Emergency banner
    if emergency:
        emerg_data = [["⚠️  EMERGENCY: Seek immediate medical attention — Call 102 / 112"]]
        emerg_tbl = Table(emerg_data, colWidths=[17*cm])
        emerg_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#fee2e2")),
            ("TEXTCOLOR",  (0,0), (-1,-1), colors.HexColor("#991b1b")),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 11),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#fee2e2")]),
            ("BOX",        (0,0), (-1,-1), 1.5, colors.HexColor("#fca5a5")),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(emerg_tbl)
        story.append(Spacer(1, 14))

    # Primary diagnosis card
    diag_data = [
        ["Primary Diagnosis", "Confidence", "Severity"],
        [disease, f"{confidence}%", severity.upper()]
    ]
    diag_tbl = Table(diag_data, colWidths=[9*cm, 4*cm, 4*cm])
    diag_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#0f172a")),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0),  10),
        ("BACKGROUND",    (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ("FONTNAME",      (0,1), (0,1),   "Helvetica-Bold"),
        ("FONTSIZE",      (0,1), (0,1),   14),
        ("FONTNAME",      (1,1), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (1,1), (1,1),   14),
        ("TEXTCOLOR",     (2,1), (2,1),   colors.HexColor(sev_color)),
        ("FONTSIZE",      (2,1), (2,1),   12),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f8fafc")]),
        ("BOX",           (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(diag_tbl)
    story.append(Spacer(1, 14))

    # Description
    story.append(Paragraph("About this condition", h2_style))
    story.append(Paragraph(description, body_style))

    # Doctor advice
    story.append(Paragraph("When to see a Doctor", h2_style))
    advice_data = [[Paragraph(f"⚕️  {doctor_advice}", warn_style)]]
    advice_tbl = Table(advice_data, colWidths=[17*cm])
    advice_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#fffbeb")),
        ("BOX",           (0,0), (-1,-1), 1, colors.HexColor("#fde68a")),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    story.append(advice_tbl)
    story.append(Spacer(1, 14))

    # Symptoms table
    if symptoms:
        story.append(Paragraph("Reported Symptoms", h2_style))
        SYM_COLOR = {"mild": "#dcfce7", "moderate": "#fef3c7", "severe": "#fee2e2"}
        SYM_TEXT  = {"mild": "#166534", "moderate": "#92400e", "severe": "#991b1b"}
        sym_rows = [["Symptom", "Severity"]]
        for s in symptoms:
            sev = sym_severities.get(s, "mild")
            sym_rows.append([s.replace("_", " ").title(), sev.upper()])
        sym_tbl = Table(sym_rows, colWidths=[12*cm, 5*cm])
        style_cmds = [
            ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#0f172a")),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ALIGN",         (1,0), (1,-1),  "CENTER"),
            ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]
        for i, row in enumerate(sym_rows[1:], 1):
            sev = row[1].lower()
            style_cmds.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor(SYM_COLOR.get(sev, "#f8fafc"))))
            style_cmds.append(("TEXTCOLOR",  (1,i), (1,i),  colors.HexColor(SYM_TEXT.get(sev, "#374151"))))
            style_cmds.append(("FONTNAME",   (1,i), (1,i),  "Helvetica-Bold"))
        sym_tbl.setStyle(TableStyle(style_cmds))
        story.append(sym_tbl)
        story.append(Spacer(1, 14))

    # Top 3
    if top3:
        story.append(Paragraph("Differential Diagnosis (Top 3)", h2_style))
        t3_rows = [["Rank", "Possible Condition", "Probability"]]
        for i, item in enumerate(top3, 1):
            t3_rows.append([f"#{i}", item["disease"], f"{item['confidence']}%"])
        t3_tbl = Table(t3_rows, colWidths=[2*cm, 11*cm, 4*cm])
        t3_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#0f172a")),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ALIGN",         (0,0), (0,-1),  "CENTER"),
            ("ALIGN",         (2,0), (2,-1),  "CENTER"),
            ("BACKGROUND",    (0,1), (-1,1),  colors.HexColor("#f0fdf4")),
            ("FONTNAME",      (0,1), (-1,1),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0,2), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
            ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        story.append(t3_tbl)
        story.append(Spacer(1, 14))

    # Precautions
    if precautions:
        story.append(Paragraph("Precautions & Self-Care", h2_style))
        prec_items = "\n".join(f"• {p}" for p in precautions)
        story.append(Paragraph(prec_items, body_style))
        story.append(Spacer(1, 10))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 8))
    disclaimer = ParagraphStyle("disc", fontSize=8, textColor=colors.HexColor("#94a3b8"),
                                 fontName="Helvetica", alignment=TA_CENTER, leading=12)
    story.append(Paragraph(
        "⚠️  This report is generated by an ML model for educational purposes only. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare professional for any health concerns.",
        disclaimer
    ))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"SymptomAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "diseases": len(le.classes_), "symptoms": len(all_symptoms)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
