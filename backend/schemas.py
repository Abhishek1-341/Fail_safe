from pydantic import BaseModel, Field
from typing import Literal, List


class Input_schema_1(BaseModel) : # this is for pre G1
    # categorical variables
    school: Literal['GP', 'MS']
    sex: Literal['F', 'M']
    address: Literal['U', 'R']
    famsize: Literal['GT3', 'LE3']
    Pstatus: Literal['A', 'T']
    Mjob: Literal['at_home', 'health', 'other', 'services', 'teacher']
    Fjob: Literal['teacher', 'other', 'services', 'health', 'at_home']
    reason: Literal['course', 'other', 'home', 'reputation']
    guardian: Literal['mother', 'father', 'other']
    schoolsup: Literal['yes', 'no']
    famsup: Literal['no', 'yes']
    paid: Literal['no', 'yes']
    activities: Literal['no', 'yes']
    nursery: Literal['yes', 'no']
    higher: Literal['yes', 'no']
    internet: Literal['no', 'yes']
    romantic: Literal['no', 'yes']
    subject: Literal['Maths', 'Portuguese']

    # numerical variables
    age: float = Field(gt=0)
    Medu: float
    Fedu: float
    traveltime: float
    studytime: float
    failures: float
    famrel: float
    freetime: float
    goout: float
    Dalc: float
    Walc: float
    health: float
    absences: float


class Input_schema_2(BaseModel) : # this is for pre G2
    # categorical variables
    school: Literal['GP', 'MS']
    sex: Literal['F', 'M']
    address: Literal['U', 'R']
    famsize: Literal['GT3', 'LE3']
    Pstatus: Literal['A', 'T']
    Mjob: Literal['at_home', 'health', 'other', 'services', 'teacher']
    Fjob: Literal['teacher', 'other', 'services', 'health', 'at_home']
    reason: Literal['course', 'other', 'home', 'reputation']
    guardian: Literal['mother', 'father', 'other']
    schoolsup: Literal['yes', 'no']
    famsup: Literal['no', 'yes']
    paid: Literal['no', 'yes']
    activities: Literal['no', 'yes']
    nursery: Literal['yes', 'no']
    higher: Literal['yes', 'no']
    internet: Literal['no', 'yes']
    romantic: Literal['no', 'yes']
    subject: Literal['Maths', 'Portuguese']

    # numerical variables
    age: float = Field(gt=0)
    Medu: float
    Fedu: float
    traveltime: float
    studytime: float
    failures: float
    famrel: float
    freetime: float
    goout: float
    Dalc: float
    Walc: float
    health: float
    absences: float
    G1: float


class Input_schema_3(BaseModel) : # this is for pre final exam
    # categorical variables
    school: Literal['GP', 'MS']
    sex: Literal['F', 'M']
    address: Literal['U', 'R']
    famsize: Literal['GT3', 'LE3']
    Pstatus: Literal['A', 'T']
    Mjob: Literal['at_home', 'health', 'other', 'services', 'teacher']
    Fjob: Literal['teacher', 'other', 'services', 'health', 'at_home']
    reason: Literal['course', 'other', 'home', 'reputation']
    guardian: Literal['mother', 'father', 'other']
    schoolsup: Literal['yes', 'no']
    famsup: Literal['no', 'yes']
    paid: Literal['no', 'yes']
    activities: Literal['no', 'yes']
    nursery: Literal['yes', 'no']
    higher: Literal['yes', 'no']
    internet: Literal['no', 'yes']
    romantic: Literal['no', 'yes']
    subject: Literal['Maths', 'Portuguese']

    # numerical variables
    age: float = Field(gt=0)
    Medu: float
    Fedu: float
    traveltime: float
    studytime: float
    failures: float
    famrel: float
    freetime: float
    goout: float
    Dalc: float
    Walc: float
    health: float
    absences: float
    G1: float
    G2: float


# the order of variable must be this
var_order = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'subject', 'G1', 'G2']

class Output_schema(BaseModel) :
    my_prediction : float
    shap_values: List
    feature_names: List
