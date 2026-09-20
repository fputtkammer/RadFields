"""
prompts and configuration variables
"""
from langchain_core.prompts import ChatPromptTemplate

# prompt templates
PROMPT_TEMPLATES = {
    "identifier": {
        "modality": ChatPromptTemplate([
            ("system", "Du bestimmst die Modalität einer radiologischen Untersuchung, deren Auswertung in einem Befundbericht ausgeführt ist. Der "
                       "Nutzer wird dir den Befundbericht übergeben, wähle dann, soweit möglich, für diesen Bericht aus den folgenden Modalitäten "
                       "die passende aus: \n {categories} \n Antworte ausschließlich im JSON Format und halte dich dafür strikt an diese Vorlage: "
                       "\n {template_read_version} \n Falls der Bericht die Ergebnisse von Untersuchungen mehrerer Modalitäten beschreibt, "
                       "gib eine der Modalitäten an. Falls im Bericht keine Angaben zur Modalität gemacht werden, trage \"{no_findings_response}\" "
                       "in die Vorlage ein. Falls keine der genannten Auswahlmöglichkeiten geeignet scheint, trage \"{other_response}\" in die "
                       "Vorlage ein."),
            ("user", "Bestimme die Modalität der Untersuchung, die dem folgenden Befundbericht zugrunde liegt: \n {report_text}")
        ]),
        "exam_type": ChatPromptTemplate([
            ("system", "Du bestimmst den Untersuchungstyp einer radiologischen Untersuchung, deren Auswertung in einem Befundbericht ausgeführt "
                       "ist. Der Nutzer wird dir den Befundbericht übergeben, wähle dann für diesen Bericht, soweit möglich, aus den folgenden "
                       "Untersuchungstypen einen passenden aus: \n {categories} \n Antworte ausschließlich im JSON Format und halte dich dafür "
                       "strikt an diese Vorlage: \n {template_read_version} \n Falls im Bericht keine Angaben zum Untersuchungstyp gemacht werden, "
                       "trage \"{no_findings_response}\" in die Vorlage ein. Falls der Bericht die Ergebnisse mehrerer Untersuchungen beschreibt, "
                       "trage \"{multiple_exams_response}\" in die Vorlage ein. Falls keine der genannten Auswahlmöglichkeiten geeignet "
                       "scheint, trage \"{other_response}\" in die Vorlage ein."),
            ("user", "Bestimme den Typ der Untersuchung, die dem folgenden Befundbericht zugrunde liegt: \n {report_text}")
        ]),
        "exam_type, multi_choice": ChatPromptTemplate([
            ("system", "Du bestimmst die Untersuchungstypen mehrerer radiologischer Untersuchungen, deren Auswertung in einem Befundbericht "
                       "ausgeführt ist. Der Nutzer wird dir den Befundbericht übergeben, wähle dann für diesen Bericht, soweit möglich, "
                       "aus den folgenden Untersuchungstypen die passenden aus: \n {categories} \n Antworte ausschließlich im JSON Format und halte "
                       "dich dafür strikt an diese Vorlage: \n {template_read_version} \n Falls der Befundbericht die Ergebnisse von Wiederholungen "
                       "der gleichen Untersuchung beschreibt, trage den gleichen Untersuchungstyp entsprechend mehrfach in die Liste ein. Falls "
                       "keine der genannten Auswahlmöglichkeiten geeignet scheint, trage  \"[\"{other_response}\"]\" in die Vorlage ein."),
            ("user", "Bestimme den Typ der Untersuchung, die dem folgenden Befundbericht zugrunde liegt: \n {report_text}")
        ]),
        "exam_type, categorize_novelties": ChatPromptTemplate([
            ("system", "Du bestimmst den Untersuchungstyp einer radiologischen Untersuchung, deren Auswertung in einem Befundbericht ausgeführt "
                       "ist. Der Nutzer wird dir den Befundbericht übergeben, wähle dann für diesen Bericht, soweit möglich, aus den folgenden "
                       "Untersuchungstypen einen passenden aus: \n {categories} \n Antworte ausschließlich im JSON Format und halte dich dafür "
                       "strikt an diese Vorlage: \n {template_read_version} \n Falls im Bericht keine Angaben zum Untersuchungstyp gemacht werden, "
                       "trage \"{no_findings_response}\" in die Vorlage ein. Falls der Bericht die Ergebnisse mehrerer Untersuchungen beschreibt, "
                       "trage \"{multiple_exams_response}\" in die Vorlage ein. Sollte lediglich keiner der aufgeführten Untersuchungstypen zu den "
                       "Angaben im Befundbericht passen, gebe eine neue Kategorie von Untersuchungstypen in deiner Antwort an. Orientiere dich "
                       "dabei bei deinem neuen Kategorienamen an den oben genannten."),
            ("user", "Bestimme den Typ der Untersuchung, die dem folgenden Befundbericht zugrunde liegt: \n {report_text}")
        ]),
        "main_diagnosis": ChatPromptTemplate([
            ("system", "Du bestimmst die Hauptdiagnose einer radiologischen Untersuchung, deren Auswertung in einem Befundbericht ausgeführt ist. "
                       "Der Nutzer wird dir den Befundbericht übergeben, wähle dann für diesen Bericht, soweit möglich, aus den folgenden "
                       "Hauptdiagnosen eine passende aus: \n {categories} \n Antworte ausschließlich im JSON Format und halte dich dafür strikt an "
                       "diese Vorlage: \n {template_read_version} \n Falls im Bericht keine Befundung vorgenommen wird, "
                       "trage \"{no_findings_response}\" in die Vorlage ein. Falls der Bericht die Ergebnisse mehrerer Untersuchungen beschreibt, "
                       "trage \"{multiple_exams_response}\" in die Vorlage ein. Falls keine der genannten Auswahlmöglichkeiten geeignet scheint, "
                       "trage \"{other_response}\" in die Vorlage ein."),
            ("user", "Bestimme die Hauptdiagnose, die im folgenden Befundbericht angegeben wird: \n {report_text}")
        ]),
        "main_diagnosis, categorize_novelties": ChatPromptTemplate([
            ("system", "Du bestimmst die Hauptdiagnose einer radiologischen Untersuchung, deren Auswertung in einem Befundbericht ausgeführt ist. "
                       "Der Nutzer wird dir den Befundbericht übergeben, wähle dann für diesen Bericht, soweit möglich, aus den folgenden "
                       "Hauptdiagnosen eine passende aus: \n {categories} \n Antworte ausschließlich im JSON Format und halte dich dafür strikt an "
                       "diese Vorlage: \n {template_read_version} \n Falls im Bericht keine Befundung vorgenommen wird, "
                       "trage \"{no_findings_response}\" in die Vorlage ein. Falls der Bericht die Ergebnisse mehrerer Untersuchungen beschreibt, "
                       "trage \"{multiple_exams_response}\" in die Vorlage ein. Sollte lediglich keine der aufgeführten Hauptdiagnosen zu den "
                       "Angaben im Befundbericht passen, gebe eine neue Kategorie von Hauptdiagnosen in deiner Antwort an. Orientiere dich dabei in "
                       "bei deinem neuen Kategorienamen an den oben genannten."),
            ("user", "Bestimme die Hauptdiagnose, die im folgenden Befundbericht angegeben wird: \n {report_text}")
        ]),
        "icd_10": ChatPromptTemplate([
            ("system", "Du bestimmst die Hauptdiagnose und den zugehörigen ICD10 Code einer radiologischen Untersuchung, deren Auswertung in einem "
                       "Befundbericht ausgeführt ist. Der Nutzer wird dir den Befundbericht übergeben und dich zu den entsprechenden Angaben "
                       "auffordern. Gebe in deiner Antwort immer als erstes kurz die Hauptdiagnose an. Halte dich dabei an die Formulierung im "
                       "Bericht. Bestimme anschließend, auf dieser Basis, den ICD10 Code. Gebe ausschließlich den Code an, ohne Erklärungen oder "
                       "Erläuterungen. Antworte ausschließlich im JSON Format und halte dich dafür strikt an diese Vorlage: \n {"
                       "template_read_version} Falls sich die Hauptdiagnose der Untersuchung auf Basis des Berichtes nicht (sicher) identifizieren "
                       "lässt, trage \"\" in beide Felder der Vorlage ein."),
            ("user", "Bestimme die Hauptdiagnose, die im folgenden Befundbericht angegeben wird, und gebe den zugehörigen ICD10 Code an: \n {"
                     "report_text}")

        ]),
        "main_indication, categorize_novelties": ChatPromptTemplate([
            ("system", "Du klassifizierst die Hauptindikation, die ausschlaggebend für eine radiologischen Untersuchung war. Der Nutzer wird dir "
                       "dazu den erstellten Befundbericht übergeben, wähle dann für diesen Bericht, soweit möglich, aus den folgenden "
                       "Hauptindikation eine passende aus: \n {categories} \n Sollte keine der aufgeführten Hauptindikationen zu den Angaben im "
                       "Befundbericht passen, gebe eine neue Kategorie von Hauptindikationen in deiner Antwort an. Orientiere dich dabei in bei "
                       "deinem neuen Kategorienamen an den oben genannten. Antworte ausschließlich im JSON Format und halte dich dafür strikt an "
                       "diese Vorlage: \n {template_read_version}"),
            ("user", "Bestimme die Hauptindikation, die im folgenden Befundbericht angegeben wird: \n {report_text}")
        ])
    },
    "generator": {
        "select_template": ChatPromptTemplate([
            ("system", "Du suchst ein passendes Template für die Strukturierung eines radiologischen Befundtextes aus. Der Nutzer wird dir zunächst "
                       "den Befundtext übergeben. Wähle dann, soweit möglich, eine der folgenden Optionen aus: \n {template_names_and_explanations} "
                       "\n Antworte ausschließlich im JSON Format und halte dich dabei an diese Vorlage: \n {response_template} \n Trage in das "
                       "Feld ausschließlich den Namen des Templates ein und verzichte auf Erläuterungen. Orientiere dich bei deiner Entscheidung an "
                       "den Erklärungen für welche Untersuchungen / Fragestellungen die Templates geeignet sind. Sollte keines der Templates zu dem "
                       "gegebenen Befundtext passen, trage \"{other_response}\" in das Feld der Antwortmaske ein. Falls im Bericht keine Befundung "
                       "vorgenommen wird, trage \"{no_findings_response}\" in die Vorlage ein."),
            ("user", "Suche ein passendes Template für den folgenden Befundbericht aus: \n {report_text}")
        ]),
        "select_template, multi_choice": ChatPromptTemplate([
            ("system", "Du suchst passende Templates für die Strukturierung eines radiologischen Befundtextes aus. Der Nutzer wird dir zunächst den "
                       "Befundtext übergeben. Wähle dann aus den folgenden Optionen maximal {n_max_templates} passende aus: \n {"
                       "template_names_and_explanations} \n Antworte ausschließlich im JSON Format und halte dich dabei an diese Vorlage: \n {"
                       "response_template} \n Trage in die Liste im Feld ausschließlich die Namen der Templates ein und verzichte auf "
                       "Erläuterungen. Orientiere dich bei deiner Entscheidung an den Erklärungen für welche Untersuchungen / Fragestellungen die "
                       "Templates geeignet sind. Sollte keines der Templates zu dem gegebenen Befundtext passen, trage [\"{other_response}\"] in "
                       "das Feld der Antwortmaske ein. Falls im Bericht keine Befundung vorgenommen wird, trage [\"{no_findings_response}\"] in die "
                       "Vorlage ein."),
            ("user", "Suche passende Templates für den folgenden Befundbericht aus: \n {report_text}")
        ]),
        "select_template, multi_choice, multiple_exams": ChatPromptTemplate([
            ("system", "Du suchst passende Templates für die Strukturierung eines radiologischen Befundtextes aus. Der Nutzer wird dir zunächst den "
                       "Befundtext übergeben. Wähle dann aus den folgenden Optionen passende aus: \n {template_names_and_explanations} \n Wähle für "
                       "jede Untersuchung, die im Bericht beschrieben ist, jeweils ein Template aus. Antworte ausschließlich im JSON Format und "
                       "halte dich dabei an diese Vorlage: \n {response_template} \n Trage in die Liste im Feld ausschließlich die Namen der "
                       "Templates ein und verzichte auf Erläuterungen. Orientiere dich bei deiner Entscheidung an den Erklärungen für welche "
                       "Untersuchungen / Fragestellungen die Templates geeignet sind. Falls mehrere der beschriebenen Untersuchung mit dem gleichen "
                       "Template strukturiert werden können, trage dieses entsprechend mehrfach in die Liste ein. Sollte keines der Templates zu "
                       "dem gegebenen Befundtext passen, trage [\"{other_response}\"] in das Feld der Antwortmaske ein."),
            ("user", "Suche passende Templates für den folgenden Befundbericht aus: \n {report_text}")
        ]),
        "structure_report": ChatPromptTemplate([
            ("system", "Du strukturierst einen radiologischen Befundbericht. Der Nutzer wird dir den Bericht übergeben und dich zu einer "
                       "strukturierten Antwort auffordern. Antworte ausschließlich im JSON format und halte dich dabei streng an dieses Template: "
                       "\n {template_reader_view} \n Extrahiere lediglich die Informationen, die für das jeweilige Feld relevant sind, "
                       "aus dem Bericht und füge keine Interpretationen oder Zusammenfassungen hinzu. Orientiere dich bei der Extraktion auch an "
                       "den Kommentaren mit denen die verschiedenen Abschnitte versehen sind. Oft wirst du zu einem der Abschnitte keine "
                       "Informationen im Befundbericht finden, trage in diesem Fall \"{information_missing_response}\" in das entsprechende Feld "
                       "ein."),
            ("user", "Strukturiere den folgenden Befundbericht: \n {report_text}")
        ]),
        "structure_report, sequentially": {
            "initial": ChatPromptTemplate(
                [("system", "Du strukturierst einen radiologischen Befundbericht, indem du mehrere Templates auf Basis der im Bericht enthaltenen "
                            "Informationen ausfüllst. Der Nutzer wird dir zunächst den Bericht übergeben und dich dann nacheinander zum Ausfüllen "
                            "eines bestimmten Templates auffordern. Dies sind die auszufüllenden Templates mit ihren jeweiligen Titeln: \n {"
                            "template_info} \n Antworte ausschließlich im JSON format und halte dich dabei streng an das vom Nutzer "
                            "referenzierte Template. Extrahiere lediglich die Informationen, die für das jeweilige Feld relevant sind, "
                            "aus dem Bericht und füge keine Interpretationen oder Zusammenfassungen hinzu. Orientiere dich bei der Extraktion auch "
                            "an den Kommentaren, mit denen die verschiedenen Abschnitte versehen sind. Oft wirst du zu einem der Abschnitte keine "
                            "Informationen im Befundbericht finden, trage in diesem Fall \"{information_missing_response}\" in das entsprechende "
                            "Feld ein."),
                 ("user", "Strukturiere den folgenden Befundbericht: \n {report_text} \n Fülle dazu zunächst das Template \"{template_title}\" aus.")]),
            "follow-up": ChatPromptTemplate(
                [("user", "Fülle nun das Template \"{template_title}\" aus.")]),
        }
    }
}

# sentinel values to be returned by LLM instead of class name
SPECIAL_RESPONSES = {
            "no_findings_response": "Keine Befundung",  # indicates report stub
            "other_response": "Keine der angegebenen Optionen",  # indicates no fitting option
            "multiple_exams_response": "Mehrere Untersuchungen beschrieben",  # indicates multiple examinations detailed in same report text
            "default_field_value": ""  # value to display in reader-view of response templates & returned if no valid response
        }