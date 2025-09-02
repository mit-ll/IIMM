from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper


FMOW_CLASSNAMES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field", "dam",
    "debris_or_rubble", "educational_institution", "electric_substation",
    "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad",
    "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
    "parking_lot_or_garage", "place_of_worship", "police_station", "port",
    "prison", "race_track", "railway_bridge", "recreational_facility",
    "road_bridge", "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm", "space_facility",
    "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
    "wind_farm", "zoo"
]


class FMoW:

    def __init__(
            self,
            preprocess,
            location='./data',
            batch_size=128,
            num_workers=16
    ):

        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.dataset = get_dataset(dataset="fmow", download=False, root_dir=location)
        self.grouper = CombinatorialGrouper(self.dataset, ["region"])
        self.classnames = FMOW_CLASSNAMES

        self.populate_train()
        self.populate_test()

    def populate_train(self):

        self.train_dataset = self.dataset.get_subset("train", transform=self.preprocess)
        self.train_loader = get_train_loader(
            "standard",
            self.train_dataset,
            uniform_over_groups=True,
            grouper=self.grouper,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            )
        
        self.id_val = None
        self.ood_val = None

    def populate_test(self):

        self.test_datasets = None
        self.test_loader = None