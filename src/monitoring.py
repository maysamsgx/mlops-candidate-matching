import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import BatchRequest
import json
import os

class DataValidator:
    """
    [Mohammed Ali]: The Guardrail.
    Using Great Expectations here. If data comes in garbage, we fail fast before hitting the model.
    """
    def __init__(self, context_root_dir="great_expectations"):
        self.context = gx.get_context(mode="ephemeral")
        self.expectation_suite_name = "candidate_data_suite"
        
    def create_expectation_suite(self, df: pd.DataFrame):
        """
        Creates a baseline expectation suite from a reference dataframe.
        """
        datasource_name = "pandas_datasource"
        data_asset_name = "candidate_data"
        
        # Add datasource if not exists
        try:
             self.context.sources.add_pandas(datasource_name)
        except:
            pass # Already exists
            
        validator = self.context.sources.pandas_default.read_dataframe(
            df, dictionary_assets=True
        )
        
        # Basic expectations
        validator.expect_column_values_to_not_be_null("candidate_id")
        validator.expect_column_values_to_not_be_null("skills")
        validator.expect_column_values_to_not_be_null("experience_level")
        
        # Categorical set expectations (Example)
        # unq_roles = df['job_role'].unique().tolist()
        # validator.expect_column_values_to_be_in_set("job_role", unq_roles)
        
        # Save suite
        validator.save_expectation_suite(discard_failed_expectations=False)
        return self.expectation_suite_name

    def validate(self, df: pd.DataFrame):
        """
        Validates new batch of data against the suite using Great Expectations.
        """
        # [Mohammed Ali]: connecting to the GE validation engine.
        # We wrap the dataframe in a GE Validator to run the checks "for real".
        
        datasource_name = "pandas_datasource"
        try:
             self.context.sources.add_pandas(datasource_name)
        except:
            pass 
            
        # Create a validator for this batch
        validator = self.context.sources.pandas_default.read_dataframe(
            df, dictionary_assets=True
        )
        
        # Define expectations on the fly for this demo (in prod, load from JSON)
        validator.expect_column_values_to_not_be_null("candidate_id")
        validator.expect_column_values_to_not_be_null("skills")
        validator.expect_column_values_to_not_be_null("experience_level")
        validator.expect_column_to_exist("qualification")
        
        # Run validation
        validation_result = validator.validate()
        
        success = validation_result["success"]
        results = []
        if not success:
            results.append("Great Expectations Validation Failed")
            # We could parse validation_result for details, but keeping it brief.
        
        return {"success": success, "results": results, "details": validation_result.to_json_dict()}

def check_drift(reference_df, current_df, threshold=0.05):
    """
    [Misem]: The CME heartbeat.
    Calculates statistical drift on the fly. If this spikes, we need to know ASAP.
    """
    # Compare distribution of 'experience_level'
    ref_dist = reference_df['experience_level'].value_counts(normalize=True)
    curr_dist = current_df['experience_level'].value_counts(normalize=True)
    
    # Align indexes
    all_cats = list(set(ref_dist.index) | set(curr_dist.index))
    drift_score = 0
    
    for cat in all_cats:
        p = ref_dist.get(cat, 0)
        q = curr_dist.get(cat, 0)
        drift_score += abs(p - q)
        
    is_drift = drift_score > threshold
    return {"drift_detected": is_drift, "score": drift_score}
