import os
import re
import tempfile
import pandas as pd
import streamlit as st
from mistralai.client import Mistral
### Developer Notes: The current model uses Mistrals genrous monthly API limits in case of any chnaged in:
#1-Usage limits: Feature enginerinbg using Pillow can be used to reduce input token and reduce the load
#2- Deployment Issues: As shon clearly  we are not using any Docker container so in case of any updates
#in packages update a small action is needed to update to lastest to keep the script working
st.title("SAS - Token Reduction-Based Model for  VIN Extrcation")

uploaded_file = st.file_uploader("UPLOAD PDF FILE HERE!", type=["pdf"])

if st.button("Start OCR", disabled=not uploaded_file):
       
    api_key = st.secrets["MISTRAL_API_KEY"]
    

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        client = Mistral(api_key=api_key)

        with st.spinner("OCR is cooking, please wait..."):
            with open(tmp_path, "rb") as f:
                uploaded = client.files.upload(
                    file={"file_name": uploaded_file.name, "content": f.read()},
                    purpose="ocr"
                )

        with st.spinner("Running OCR..."):
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "file", "file_id": uploaded.id},
                include_image_base64=False
            )

        pages = ocr_response.pages
        returned_nums = {p.index + 1 for p in pages}

        skipped = set()
        if returned_nums:
            skipped = set(range(1, max(returned_nums) + 1)) - returned_nums
            if skipped:
                st.warning(f"Pages skipped by OCR: {sorted(skipped)} — check these manually.")

        vin_pattern = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b')
        records = []

        for page_data in pages:
            page_num = page_data.index + 1
            text = page_data.markdown.upper()
            valid = [v for v in vin_pattern.findall(text) if not v.isdigit() and not v.isalpha()]

            if valid:
                for vin in valid:
                    records.append({
                        "VIN": vin,
                        "Source_Page": page_num,
                        "File_Name": uploaded_file.name,
                        "Page_Status": "VIN_FOUND"
                    })
            else:
                records.append({
                    "File_Name": uploaded_file.name,
                    "VIN": None,
                    "Source_Page": page_num,
                    "Page_Status": "NO_VIN_DETECTED"
                })

        for sp in sorted(skipped):
            records.append({
                "File_Name": uploaded_file.name,
                "VIN": None,
                "Source_Page": sp,
                "Page_Status": "PAGE_SKIPPED_BY_OCR"
            })

        df = pd.DataFrame(records).sort_values("Source_Page").reset_index(drop=True)
        vin_df = df[df["VIN"].notna()].drop_duplicates(subset=["VIN"]).reset_index(drop=True)
        final_df = pd.concat([vin_df, df[df["VIN"].isna()]], ignore_index=True)

        st.success(f"Done — {len(vin_df)} unique VINs extracted.")

        col1, col2, col3 = st.columns(3)
        col1.metric("VINs Found", len(vin_df))
        col2.metric("Empty Pages", int((final_df["Page_Status"] == "NO_VIN_DETECTED").sum()))
        col3.metric("Pages Skipped by OCR", int((final_df["Page_Status"] == "PAGE_SKIPPED_BY_OCR").sum()))

        st.dataframe(vin_df, use_container_width=True)

        st.download_button("Download Output as CSV", data=final_df.to_csv(index=False).encode("utf-8"),
                           file_name="VIN_OUTPUT.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        os.unlink(tmp_path)