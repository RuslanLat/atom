import re
import time as my_time
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt


# задание палитры цветов
colors = ["#09214a", "#567189", "#7B8FA1", "#CFB997", "#FAD6A5"]

st.set_page_config(
    page_title="Атом - ЛИФТ",
    page_icon="streamlit_app/images/atom.png",
    layout="wide",
)  # layout = "wide"

# Загрузка модели и токенизатора BART
model = BartForConditionalGeneration.from_pretrained("./streamlit_app/bart_model")
tokenizer = BartTokenizer.from_pretrained("./streamlit_app/bart_model")

# загрузка берт с локального компа
model_bert = SentenceTransformer("./streamlit_app/roberta_model")


def generate_difference(row):
    text1 = row["Text_uc"]
    text2 = row["Text_ssts"]

    # Форматирование входных данных для модели BART
    input_text = (
        f"Identify the different content, text1 and text2: "
        f"First text: {text1} </s> Second text: {text2}"
    )

    # input_text = f"Find the differences in the second text compared to the first: {text2} </s> {text1}"
    # input_text = (
    #     f"Context the differences in characteristics and specifications between the first text and the second text. "
    #     f"First text: {text1} </s> Second text: {text2}"
    # )
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Проверка длины входных данных на размерность токена 1024
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]

    # Генерация описания расхождений
    if text2 != "-":
        try:
            outputs = model.generate(
                input_ids,
                max_length=60,
                num_beams=7,
                early_stopping=True,
                do_sample=True,
                temperature=0.95,
                top_k=50,
                top_p=0.90,
            )
            description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except IndexError as e:
            print(f"Error during generation: {e}")
            return "Ошибка генерации"
        description = ":".join(description.split(":")[2:])
    else:
        description = "-"
    return description


def generate_description(row):

    text1 = row["Text_uc"]
    text2 = row["Text_ssts"]

    # Форматирование входных данных для модели BART
    input_text = (
        f"Discribe the differences in characteristics and specifications between the first text and the second text. "
        f"First text: {text1} </s> Second text: {text2}"
    )
    # input_text = f"Find the differences in the first text compared to the second: {text1} </s> {text2}"
    # input_text = f"Find the differences in the first text compared to the second that are not in the third: {text1} </s> {text2} </s> {text3}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Проверка длины входных данных на размерность токена 1024
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]

    # Генерация описания расхождений
    try:
        outputs = model.generate(
            input_ids,
            max_length=150,
            num_beams=5,
            early_stopping=True,
            do_sample=True,
            temperature=0.65,
            top_k=50,
            top_p=0.90,
        )
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except IndexError as e:
        print(f"Error during generation: {e}")
        return "Ошибка генерации"
    description = ":".join(description.split(":")[2:])
    return description


# Загрузка модели DistilBERT


def preprocess_text(text):
    if text == "-":  # Проверка на NaN
        return "-"
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r"\s+", " ", text)  # Удаление лишних пробелов
    text = re.sub(r"[^\w\s]", "", text)  # Удаление пунктуации
    return text


# Функция для вычисления косинусного сходства
def compute_cosine_similarity(row):
    text1 = preprocess_text(row["Text_uc"])
    text2 = preprocess_text(row["Difference"])

    # Проверка на наличие NaN значений
    if text1 == "-" or text2 == "-":
        return 0

    # Получение векторных представлений для текстов
    embeddings1 = model_bert.encode(text1)
    embeddings2 = model_bert.encode(text2)

    # Вычисление косинусного сходства
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

    return cosine_similarity.item()


def open_docx_file(pacth, file_folder: str) -> List[str]:
    """функция извлечения текста из файлов с расширением .docx

    Args:
        pacth (str): путь к файлу

    Returns:
        List[str]: [наименование функции, текс файла]
    """

    document = Document(pacth)
    paragraphs = [
        p.text.strip().replace("\xa0", " ")
        for p in document.paragraphs
        if len(p.text) != 0
    ]

    if file_folder == "HMI":
        return [paragraphs[0].split("]")[-1].strip(), " ".join(paragraphs[1:])]
    if file_folder == "SSTS":
        return [paragraphs[0].strip(), " ".join(paragraphs[1:])]


def findall_numbers(text: str) -> int:
    """функция извлечения номера файла из patch файла

    Args:
        text (str): путь к файлу

    Returns:
        int: номер файла
    """
    numbers = re.findall(r"\b\d+\b", text)
    return numbers[0]


def file_processing(
    list_files: List[str], directory_name: str = "HMI"
) -> List[List[str]]:
    """функция извлечени ключевых призноков из файла

    Args:
        phmi_files: List[str]: список путей к файлам
        directory_name: наименование директории с файлами

    Returns:
        List[Lst[str]]: признаки данных
    """

    # обработка файлов
    text_list = []
    for file in list_files:
        name, text = open_docx_file(file, directory_name)
        number = findall_numbers(file.name)
        # формирование признаков
        text_list.append([number, name, text])

    return text_list


def make_df(text: List[str]) -> pd.DataFrame:
    """функция преобразования текста в Data Frame

    Args:
        text (List[str]): признаки данных файлов

    Returns:
        pd.DataFrame: Data Frame с данными призноков по каждому файлу
    """

    df = pd.DataFrame(text, columns=["Number", "Name", "Text"])

    return df


def merge_df(hmi_df: pd.DataFrame, ssts_df: pd.DataFrame) -> pd.DataFrame:
    """функция объединения признаков данных

    Args:
        hmi_df (pd.DataFrame): Data Frame с данными призноков файлов HMI
        ssts_df (pd.DataFrame): Data Frame с данными призноков файлов SSTS

    Returns:
        pd.DataFrame: признаки данных по документации
    """

    df = hmi_df.merge(ssts_df, on="Number", how="left", suffixes=["_uc", "_ssts"])

    df = df.fillna("-")

    return df


def complience_level(value) -> str:
    """функция интерпритации соответствия

    Args:
        value (_type_): показатель соответствия

    Returns:
        str: метка класса
    """
    if value > 0.83:
        return "FC"
    elif value > 0.75 and value <= 0.83:
        return "LC"
    elif 0.01 < value <= 0.75:
        return "NC"
    else:
        return "NA"


# визуализация соотношения оценок
def func(pct, allvals):
    absolute = float(pct / 100.0 * np.sum(allvals))
    return "{:.1f}% ({:.2f})".format(pct, absolute)


col1, col2 = st.columns([1, 5])
col1.markdown(
    """<p><svg width="41" height="40" viewBox="0 0 41 40" fill="none"><path d="M20 40H20.0806C31.129 40 40.0806 31.0887 40.0806 20C40.0806 8.91129 31.1694 0 20.0806 0H20C8.91129 0 0 8.91129 0 20C0 31.0887 8.95161 40 20 40Z" fill="#00FFFF"></path><path d="M20.0194 37.9484H20.0597C31.2694 37.9484 38.9307 30.6097 38.9307 20.7307C38.9307 10.8517 30.5436 1.25488 20.0597 1.25488H20.0194C9.53553 1.25488 1.14844 10.4081 1.14844 20.7307C1.14844 31.0533 8.80973 37.9484 20.0194 37.9484Z" fill="#00EDED"></path><path d="M20.8482 35.9167C33.1869 35.9167 37.7837 29.2231 37.7837 21.4005C37.7837 10.6747 29.1143 2.48926 20.0417 2.48926C10.9692 2.48926 2.2998 10.6747 2.2998 21.4005C2.2998 29.2231 6.89658 35.9167 19.2353 35.9167H20.8482Z" fill="#00DCDC"></path><path d="M23.1458 33.8641H16.9361C6.41189 33.8641 3.3877 28.219 3.3877 22.2512C3.3877 12.8964 11.2909 3.74316 20.0006 3.74316H20.0812C28.7909 3.74316 36.6941 12.8964 36.6941 22.2512C36.6941 28.219 33.67 33.8641 23.1458 33.8641Z" fill="#00CACA"></path><path d="M12.4988 31.9685C15.0794 31.9685 17.4585 31.8072 19.9988 31.8072H20.0794C22.6198 31.8072 24.9988 31.9685 27.5794 31.9685C32.8214 31.9685 35.6036 27.6943 35.6036 22.412C35.6036 14.7507 27.7811 4.99268 20.0794 4.99268H19.9988C12.2972 4.99268 4.47461 14.7507 4.47461 22.412C4.47461 27.6943 7.25687 31.9685 12.4988 31.9685Z" fill="#00B8B8"></path><path d="M11.8154 30.6364C14.0332 30.6364 16.4928 29.7896 20.0009 29.7896H20.0816C23.5896 29.7896 26.0493 30.6364 28.267 30.6364C32.5009 30.6364 34.517 27.1686 34.517 23.1767C34.517 19.1848 32.2187 14.8299 29.759 12.0073C27.4606 9.34605 24.2751 6.24121 20.0816 6.24121H20.0009C15.8074 6.24121 12.6219 9.34605 10.3235 12.0073C7.86382 14.8299 5.56543 19.0235 5.56543 23.1767C5.56543 27.3299 7.58156 30.6364 11.8154 30.6364Z" fill="#00A6A6"></path><path d="M11.0478 29.428C13.5881 29.428 16.2494 27.6538 19.9994 27.6538H20.08C23.83 27.6538 26.4913 29.428 29.0317 29.428C31.7736 29.428 33.4671 26.7667 33.4671 23.5812C33.4671 19.7909 30.9671 15.4763 28.4671 12.6538C25.8462 9.71021 23.3865 7.45215 20.08 7.45215H19.9994C16.6929 7.45215 14.2333 9.71021 11.6123 12.6538C9.1123 15.4763 6.6123 19.7909 6.6123 23.5812C6.6123 26.7667 8.30585 29.428 11.0478 29.428Z" fill="#009494"></path><path d="M11.07 28.4152C13.4087 28.4152 16.0297 25.6733 19.9006 25.6733H20.1829C24.0539 25.6733 26.6748 28.4152 29.0135 28.4152C31.07 28.4152 32.441 26.3588 32.441 24.1007C32.441 21.0765 30.0216 17.0039 28.0861 14.6249C25.949 11.9636 23.2474 8.73779 20.1829 8.73779H19.9006C16.8361 8.73779 14.1345 11.9636 11.9974 14.6249C10.0619 17.0039 7.64258 21.0765 7.64258 24.1007C7.64258 26.3588 9.01355 28.4152 11.07 28.4152Z" fill="#008383"></path><path d="M19.8779 23.6469C17.7812 23.6469 15.9666 24.8969 14.2731 25.9856C12.7408 26.9533 11.9747 27.3969 11.1279 27.3969C9.71664 27.3969 8.62793 26.1872 8.62793 24.4533C8.62793 21.5501 11.7731 17.3162 13.5473 15.0179C15.2005 12.8808 17.257 9.97754 19.9183 9.97754H20.1602C22.8215 9.97754 24.8779 12.8808 26.5312 15.0179C28.3053 17.3162 31.4505 21.5501 31.4505 24.4533C31.4505 26.1872 30.3618 27.3969 28.9505 27.3969C28.1037 27.3969 27.3376 26.9533 25.8053 25.9856C24.1118 24.8969 22.2973 23.6469 20.2005 23.6469H19.8779Z" fill="#007171"></path><path d="M20.0415 11.2354C18.227 11.2354 17.3399 12.3241 15.7673 14.4208C13.8318 17.0418 11.8963 19.9047 10.6463 21.8402C9.75923 23.2515 9.55762 23.937 9.55762 24.5015C9.55762 25.6708 10.3641 26.3966 11.4125 26.3966C12.3399 26.3966 13.0254 25.8321 14.356 24.7434C16.1705 23.2112 17.9044 21.5983 20.0415 21.5983C22.1786 21.5983 23.9125 23.2112 25.727 24.7434C27.0576 25.8321 27.7431 26.3966 28.6705 26.3966C29.7189 26.3966 30.5254 25.6708 30.5254 24.5015C30.5254 23.937 30.3237 23.2515 29.4366 21.8402C28.1866 19.9047 26.2512 17.0418 24.3157 14.4208C22.7431 12.3241 21.856 11.2354 20.0415 11.2354Z" fill="black"></path></svg></p>""",
    unsafe_allow_html=True,
)
col2.markdown(
    '<p style="text-align: center;"><svg width="107" height="16" viewBox="0 0 107 16" fill="none"><path d="M18.3523 1.67338C17.7954 0.850566 17.173 0.561523 16.2384 0.561523H12.1205C11.1859 0.561523 10.5635 0.850566 10.0066 1.67338L0.859375 15.2064H4.42038L5.97738 12.9576C6.35506 12.4007 6.77898 12.1117 7.51315 12.1117H20.8438C21.578 12.1117 22.0019 12.4007 22.3796 12.9576L23.9366 15.2064H27.4976L18.3523 1.67338ZM11.6985 4.32101C12.0319 3.80844 12.4327 3.38644 13.2555 3.38644H15.1034C15.9262 3.38644 16.3058 3.80844 16.6604 4.32101L19.9362 9.14996H8.42074L11.6966 4.32101H11.6985Z" fill="#FFFFFF"></path><path d="M55.3766 8.57399V7.19429C55.3766 5.28082 55.1993 4.52353 57.134 3.94544C58.9588 3.41168 62.6084 3.23247 63.834 3.23247C65.0595 3.23247 68.7073 3.40975 70.5321 3.94544C72.4687 4.52353 72.2895 5.28082 72.2895 7.19429V8.57399C72.2895 10.4875 72.4667 11.2447 70.5321 11.8228C68.7073 12.3566 65.0576 12.5358 63.834 12.5358C62.6104 12.5358 58.9588 12.3585 57.134 11.8228C55.1974 11.2447 55.3766 10.4875 55.3766 8.57399ZM52.3281 5.21338V10.5549C52.3281 12.6244 52.5729 13.6033 54.7311 14.4493C56.6233 15.1834 60.273 15.696 63.834 15.696C67.395 15.696 71.0446 15.1834 72.9369 14.4493C75.0951 13.6033 75.3398 12.6244 75.3398 10.5549V5.21338C75.3398 3.14383 75.0951 2.16494 72.9369 1.31901C71.0446 0.584836 67.395 0.0722656 63.834 0.0722656C60.273 0.0722656 56.6233 0.584836 54.7311 1.31901C52.5729 2.16494 52.3281 3.14383 52.3281 5.21338Z" fill="#FFFFFF"></path><path d="M35.6924 15.2064H38.6753V4.52337C38.6753 3.67743 39.0318 3.38839 39.8334 3.38839H49.2485L48.3814 1.58669C48.0037 0.785076 47.468 0.563477 46.5777 0.563477H27.7938C26.9035 0.563477 26.3698 0.785076 25.9902 1.58669L25.123 3.38839H34.5381C35.3398 3.38839 35.6962 3.67743 35.6962 4.52337V15.2064H35.6924Z" fill="#FFFFFF"></path><path d="M92.4504 10.3774H94.0093C95.0556 10.3774 95.522 10.0441 95.9228 9.50837L101.53 1.9181C102.264 0.916083 102.754 0.561523 103.956 0.561523H106.07V15.2064H103.087V4.56766L97.368 12.4682C96.7668 13.291 96.0769 13.6263 95.1192 13.6263H91.3809C90.4232 13.6263 89.7122 13.2929 89.111 12.4682L83.3687 4.58886V15.2044H80.3857V0.561523H82.4996C83.702 0.561523 84.2127 0.96233 84.9256 1.9181L90.535 9.50837C90.9358 10.0421 91.4021 10.3774 92.4485 10.3774H92.4504Z" fill="#FFFFFF"></path></svg></p>',
    unsafe_allow_html=True,
)
col2.markdown(
    "<p style='text-align: center; color: blac;'> Контроль и управление изменениями в тендерных закупках </p>",
    unsafe_allow_html=True,
)

st.write("##")


col1, col2, col3 = st.columns(3)
with col2:
    atom_form = st.form("atom")
    uploaded_file_hmi = atom_form.file_uploader(
        "Загрузка файлов HMI",
        type=["docx"],
        accept_multiple_files=True,
        help="загрузите файлы HMI",
        key="HMI",
    )
    uploaded_file_sstm = atom_form.file_uploader(
        "Загрузка файлов SSTM",
        type=["docx"],
        accept_multiple_files=True,
        help="загрузите файлы SSTM",
        key="SSTM",
    )
    submitted = atom_form.form_submit_button("Сформировать отчёт", type="primary")
    if submitted and uploaded_file_hmi and uploaded_file_sstm:
        st.success("Файлы загружены", icon="✅")
    elif submitted and not uploaded_file_hmi or submitted and uploaded_file_sstm:
        st.error("Загрузите файлы", icon="❌")
    else:
        st.warning("Загрузите файлы", icon="⚠️")

if submitted and uploaded_file_hmi and uploaded_file_sstm:
    start = my_time.time()  ## точка отсчета времени
    # начало обработки
    hmi_text = file_processing(uploaded_file_hmi, "HMI")
    ssts_text = file_processing(uploaded_file_sstm, "SSTS")
    hmi_df = make_df(hmi_text)
    # удаление не релевантного текста
    hmi_df["Text"] = hmi_df["Text"].apply(
        lambda x: " ".join(x.split()[1:]) if "Description" in x else x
    )
    ssts_df = make_df(ssts_text)
    # удаление не релевантного текста
    ssts_df["Text"] = ssts_df["Text"].apply(
        lambda x: " ".join(x.split()[2:]) if "Functional Description" in x else x
    )
    df = merge_df(hmi_df, ssts_df)

    # получения различий в контексте текстов 2 документов
    df["Difference"] = df.apply(generate_difference, axis=1)
    # Применение функции к каждой строке DataFrame
    df["Description"] = df.apply(generate_description, axis=1)
    df["cosine_similarity"] = df.apply(compute_cosine_similarity, axis=1)
    # оценка соответствия
    df["len"] = df["Difference"].map(lambda x: len(x))
    df["len_worlds"] = df["Difference"].map(lambda x: len(x.split()))
    df["total_score"] = df.apply(
        lambda x: (
            (x["len_worlds"] / x["len"] * 0.1) + (x["cosine_similarity"] * 0.9)
            if x["Text_ssts"] != "-"
            else 0
        ),
        axis=1,
    )
    # применение функции
    df["Complience Level"] = df["total_score"].map(complience_level)
    # формирование файла отчёта
    sub = df[["Number", "Name_uc", "Difference", "Description", "Complience Level"]]
    sub = sub.rename(columns={"Name_uc": "Name"})
    # завершение обработки файлов
    end = my_time.time() - start  ## собственно время работы программы

    st.write("### Результаты работы алгоритма")
    st.write(
        f"""
    ✔️ Количество обработанных файлов:  {len(uploaded_file_hmi) + len(uploaded_file_sstm)} 

    ✔️ Время работы алгоритма:  {str(round(end, 2)) + " секунд(ы)"}

    ✔️ Среднее время обработки одного файла:  {str(round(end/(len(uploaded_file_hmi) + len(uploaded_file_sstm)), 2)) + " секунд(ы)"}
    """
    )
    st.divider()
    st.write("### Отчёт")
    st.write(sub)

    # подсчет количества оценок
    cl_count = sub["Complience Level"].value_counts()
    # наименование классов
    cl_count.index.to_list()
    # упорядочивание классов по значимости
    col_names = [
        col_name
        for col_name in ["FC", "LC", "PC", "NC", "NA"]
        if col_name in cl_count.index.to_list()
    ]
    cl_count = cl_count[col_names]

    fig, ax = plt.subplots()

    ax.pie(
        cl_count,
        colors=colors[: len(col_names)],
        autopct=lambda pct: func(pct, cl_count),
        radius=2,
        startangle=270,
        explode=[0.1, 0, 0.1, 0.1, 0.05][: len(col_names)],
    )
    ax.legend(
        labels=cl_count.index,
        title="Complience Level",
        loc="upper right",
        bbox_to_anchor=(1.5, 0, 0.5, 1),
    )
    ax.set_title("RFI/RFP result", size=20, pad=100)

    col1, col2, col3 = st.columns([1, 2, 1])
    col2.pyplot(fig, use_container_width=False)

st.write("##")
st.markdown(
    '<h5 style="text-align: center; color: blac;"> ©️ Команда "ЛИФТ" </h5>',
    unsafe_allow_html=True,
)
st.markdown(
    "<h5 style='text-align: center; color: blac;'> Цифровой прорыв 2024, Калининград </h5>",
    unsafe_allow_html=True,
)
