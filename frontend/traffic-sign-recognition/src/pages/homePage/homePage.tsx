import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.css';
import config from '../../config/config.json';
import './homePage.css';


const HomePage: React.FC = () => {
    const [selectedImages, setSelectedImages] = useState<File[]>([]);
    const [imageUrls, setImageUrls] = useState<string[]>([]);
    const [predictions, setPredictions] = useState<number[]>([]);

    const backendPredictUrl = config.backendUrl + '/' + config.backendPredictEndpoint;

    const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            const files = Array.from(event.target.files);
            setSelectedImages(files);
            setImageUrls(files.map(file => URL.createObjectURL(file)));
            setPredictions([]); // Reset predictions
        }
    };

    const handleSubmit = (event: React.FormEvent) => {
        event.preventDefault();

        const formData = new FormData();
        selectedImages.forEach((image) => {
            formData.append('image', image);
        });

        fetch(backendPredictUrl, {
            method: 'POST',
            body: formData,
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(response.statusText);
                }
                return response.json();
            })
            .then(data => {
                if (data.predictions && Array.isArray(data.predictions)) {
                    const maxIndices = data.predictions.map((list: number[]) => {
                        return list.indexOf(Math.max(...list));
                    });
                    setPredictions(maxIndices);
                } else {
                    console.error('Invalid prediction data:', data.predictions);
                }
            })
            .catch(error => {
                console.error(error);
            });
    };

    return (
        <div className="container">
            <div className="row justify-content-center">
                <div className="col-6">
                    <h1 className="text-center my-4">German Sign Recognition</h1>
                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <input type="file" accept="image/*" onChange={handleImageChange} multiple className="form-control" />
                        </div>
                        <button type="submit" className="btn btn-primary">Submit</button>
                    </form>
                    <div className="row">
                        {imageUrls.map((url, index) => (
                            <div className="col-4 mt-4" key={index}>
                                <img src={url} alt="Selected" className="img-thumbnail same-size" />

                                {predictions[index] !== undefined && <p>
                                    Prediction: {config.classeLabels[predictions[index].toString() as keyof typeof config.classeLabels]}
                                </p>}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HomePage;