# Backend/app/models/latent_models.py
# ---------------------------
# LatentPoint: Stores latent-space vectors for each input sequence snapshot.

from Backend.app import db
from datetime import datetime

class LatentPoint(db.Model):
    __tablename__ = 'latent_point'

    id = db.Column(db.Integer, primary_key=True)
    model_run_id = db.Column(db.Integer, db.ForeignKey('model_run.id'), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    group_id = db.Column(db.Integer, nullable=False)
    lag = db.Column(db.Integer, nullable=False)
    latent_vector = db.Column(db.ARRAY(db.Float), nullable=True)

    __table_args__ = (
        db.UniqueConstraint("model_run_id", "start_date", "group_id", "lag", name="uq_latent_seq"),
    )

    def get_pca_vector(self, session, config_id=None):
        """
        Returns the PCA coordinates for this latent point.
        Used to render PCA scatter plots in the frontend.
        """
        query = session.query(PCAProjection).filter_by(latent_point_id=self.id)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        return [p.value for p in query.order_by(PCAProjection.dim).all()]

    def get_som_coordinates(self, session, config_id=None):
        """
        Returns the SOM grid coordinates for this latent point.
        Used to render the SOM heatmap/clusters.
        """
        query = session.query(SOMProjection).filter_by(latent_point_id=self.id)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        return query.first()

# LatentInput: Links each latent point back to its original data source.
class LatentInput(db.Model):
    __tablename__ = 'latent_input'

    id = db.Column(db.Integer, primary_key=True)
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id'), nullable=False)
    group_id = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False)
    source_table = db.Column(db.String(80), nullable=False)

    latent_point = db.relationship('LatentPoint', backref=db.backref('inputs', lazy=True))

    def __repr__(self):
        return f"<LatentInput point={self.latent_point_id} group={self.group_id} date={self.date}>"

# SOMProjectionConfig: Stores configuration details for SOM projections.
class SOMProjectionConfig(db.Model):
    __tablename__ = 'som_projection_config'

    id = db.Column(db.Integer, primary_key=True)
    x_dim = db.Column(db.Integer)
    y_dim = db.Column(db.Integer)
    iterations = db.Column(db.Integer)
    additional_params = db.Column(db.JSON, nullable=True)

    def __repr__(self):
        # Helps identify grid size and iteration count
        return f'<SOMConfig {self.x_dim}x{self.y_dim} iters={self.iterations}>'

# SOMProjection: Maps latent points onto a 2D SOM grid cell.
class SOMProjection(db.Model):
    __tablename__ = 'som_projection'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_run_id = db.Column(db.Integer, db.ForeignKey('model_run.id'))
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id'))
    config_id = db.Column(db.Integer, db.ForeignKey('som_projection_config.id'))
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)

    @classmethod
    def get_points_for_cell(cls, session, model_run_id, x, y, config_id=None):
        """
        Retrieves all latent points assigned to a specific SOM cell.
        Powers click interactions on SOM visualizations.
        """
        query = session.query(cls).filter_by(model_run_id=model_run_id, x=x, y=y)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        return query.all()

# PCAProjectionConfig: Configuration details for PCA projections.
class PCAProjectionConfig(db.Model):
    __tablename__ = 'pca_projection_config'

    id = db.Column(db.Integer, primary_key=True)
    n_components = db.Column(db.Integer)
    additional_params = db.Column(db.JSON, nullable=True)

    def __repr__(self):
        return f'<PCAConfig components={self.n_components}>'

# PCAProjection: Stores individual PCA coordinate values for latent points.
class PCAProjection(db.Model):
    __tablename__ = 'pca_projection'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_run_id = db.Column(db.Integer, db.ForeignKey('model_run.id'))
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id'))
    config_id = db.Column(db.Integer, db.ForeignKey('pca_projection_config.id'))
    dim = db.Column(db.Integer)
    value = db.Column(db.Float)

    @classmethod
    def get_vectors_for_model(cls, session, model_run_id, config_id=None):
        """
        Aggregates PCA values into vectors per latent point.
        Used to draw PCA scatter plots for a given run and config.
        """
        query = session.query(cls).filter_by(model_run_id=model_run_id)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        results = query.order_by(cls.latent_point_id, cls.dim).all()
        vectors = {}
        for p in results:
            vectors.setdefault(p.latent_point_id, []).append(p.value)
        return vectors
