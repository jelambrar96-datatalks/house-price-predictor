import requests

url= "http://127.0.0.1:8000/api/predict"

request_data = {
    "ms_sub_class": 60, "ms_zoning": "RL", "lot_area": 10237, "street": "Pave",
    "lot_shape": "Reg", "land_contour": "Lvl", "utilities": "AllPub",
    "lot_config": "Inside", "land_slope": "Gtl", "neighborhood": "Gilbert",
    "condition1": "RRAn", "condition2": "Norm", "bldg_type": "1Fam",
    "house_style": "2Story", "overall_qual": 6, "overall_cond": 5,
    "year_built": 2005, "year_remod_add": 2007, "roof_style": "Gable",
    "roof_matl": "CompShg", "exterior1st": "VinylSd", "exterior2nd": "VinylSd",
    "mas_vnr_area": 0.0, "exter_qual": "Gd", "exter_cond": "TA",
    "foundation": "PConc", "bsmt_qual": "Gd", "bsmt_cond": "TA",
    "bsmt_exposure": "No", "bsmt_fin_type1": "Unf", "bsmt_fin_sf1": 0,
    "bsmt_fin_type2": "Unf", "bsmt_fin_sf2": 0, "bsmt_unf_sf": 783,
    "total_bsmt_sf": 783, "heating": "GasA", "heating_qc": "Ex", "central_air": "Y",
    "electrical": "SBrkr", "1st_flr_sf": 783, "2nd_flr_sf": 701, "low_qual_fin_sf": 0,
    "gr_liv_area": 1484, "bsmt_full_bath": 0, "bsmt_half_bath": 0, "full_bath": 2,
    "half_bath": 1, "bedroom_abv_gr": 3, "kitchen_abv_gr": 1, "kitchen_qual": "Gd",
    "tot_rms_abv_grd": 8, "functional": "Typ", "fireplaces": 1, "garage_type": "Attchd",
    "garage_yr_blt": 2005.0, "garage_finish": "Fin", "garage_cars": 2, "garage_area": 393,
    "garage_qual": "TA", "garage_cond": "TA", "paved_drive": "Y", "wood_deck_sf": 0,
    "open_porch_sf": 72, "enclosed_porch": 0, "3_ssn_porch": 0, "screen_porch": 0,
    "pool_area": 0, "misc_val": 0, "mo_sold": 7, "yr_sold": 2007, "sale_type": "New",
    "sale_condition": "Partial"
}


response = requests.post(url, json={"data": request_data}).json()
print(response)
