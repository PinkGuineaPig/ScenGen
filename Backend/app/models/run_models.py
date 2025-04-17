# Backend/app/models/run_models.py
# ---------------------------
# ModelRunConfig: Stores hyperparameters chosen by the user before training.

from Backend.app import db
from datetime import datetime

class ModelRunConfig(db.Model):
    __tablename__ = 'model_run_config'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_type = db.Column(db.String(50))
    parameters = db.Column(db.JSON)

    def __repr__(self):
        return f'<ModelRunConfig {self.model_type}>'


# ModelRun: Persists the trained model object and metadata.
class ModelRun(db.Model):
    __tablename__ = 'model_run'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    config_id = db.Column(db.Integer, db.ForeignKey('model_run_config.id'))
    config = db.relationship('ModelRunConfig')
    group_ids = db.Column(db.ARRAY(db.Integer))
    version = db.Column(db.Integer)
    model_blob = db.Column(db.LargeBinary)

    def __repr__(self):
        # Shows run ID and version for easy referencing
        return f'<ModelRun id={self.id} v={self.version}>'

# ModelLossHistory: Records loss/error for each epoch of a model run.
class ModelLossHistory(db.Model):
    __tablename__ = 'model_loss_history'

    id = db.Column(db.Integer, primary_key=True)
    model_run_id = db.Column(db.Integer, db.ForeignKey('model_run.id'), nullable=False)
    model_run = db.relationship('ModelRun', backref=db.backref('loss_history', lazy=True))
    epoch = db.Column(db.Integer, nullable=False)
    loss_type = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Float, nullable=False)

    def __repr__(self):
        # Useful for debugging training curves
        return f'<LossHistory run={self.model_run_id} epoch={self.epoch} type={self.loss_type}>'
