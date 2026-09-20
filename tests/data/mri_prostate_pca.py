from pydantic import BaseModel


class Technik(BaseModel):
	geraet: str = ""
	feldstaerke_in_tesla: str = ""
	sequenzen: list[str] = ["",]
	kontrastmittel: str = ""
	zusatzmedikation: str = ""
	bildqualitaet: str = ""


class Laesionen(BaseModel):
	laesion_id: str = ""
	lokalisation: str = ""
	groesse_mm: str = ""
	t2w_pirads_befund: str = ""
	dwi_pirads_befund: str = ""
	dce_pirads_befund: str = ""
	kapselbeteiligung: str = ""
	pirads_gesamt: str = ""


class Befund(BaseModel):
	prostatavolumen: str = ""
	prostatagroesse: str = ""
	haemorrhagien: str = ""
	morphologie_periphere_zone: str = ""
	morphologie_transitionalzone: str = ""
	laesionen: list[Laesionen] = [Laesionen()]
	lymphknoten: str = ""
	knochen: str = ""
	organe_kleines_becken: str = ""
	zusatzbefund: str = ""


class Pirads(BaseModel):
	laesion_id: str = ""
	lokalisation: str = ""
	pirads_score: str = ""


class Beurteilung(BaseModel):
	pirads: list[Pirads] = [Pirads()]
	gesamtbeurteilung: str = ""


class MRIProstatePCaTemplate(BaseModel):
	untersuchung: str = ""
	datum: str = ""
	klinische_angaben: str = ""
	fragestellung: str = ""
	psa: str = ""
	mr_psa_dichte: str = ""
	voraufnahmen: str = ""
	vergleich: str = ""
	technik: Technik = Technik()
	befund: Befund = Befund()
	beurteilung: Beurteilung = Beurteilung()


