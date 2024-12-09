import React, { useState } from "react";

import "./App.css";
import convertPdfToImagesAsync from "./pdf.js";

const Feedback = ({ feedback, handleFeedback, index }) => {
   return(!feedback[index.toString()] ? <p className="feedback">Do you agree with that result ?
   <button onClick={()=> handleFeedback(index)} type="button">Yes</button><button  onClick={()=> handleFeedback(index)} type="button">No</button>
   </p>: <p className="feedback">Thank you for your feedback</p>);
}

function App() {
  const [files, setFiles] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [feedback, setFeedback] = useState({});

  const UPLOAD_ENDPOINT =
    "http://degradai-api.northeurope.azurecontainer.io:8000/infer/degradai";

  const uploadFile = async file => {
    const formData = new FormData();
    formData.append("image", file);

    const response = fetch(UPLOAD_ENDPOINT, {
      method: 'POST',
      body:formData,
      mode: 'cors'
    });
    const o = await response;
    const u = await o.json();
    if (u.mention)
      return [u.mention];
    return [u.classname];
  };

  const handleOnChange = async e => {
    const file = e.target.files[0];
    e.target.value = null;

    const objectURL = URL.createObjectURL(file);
    setFiles([objectURL]);

    setPredictions(null);
    setFeedback({});
    let res = await uploadFile(file);
    setPredictions(res)
  };

  const handleFeedback = index => {
    setFeedback({...feedback, [index.toString()]: true});
  };

  function handleClick() {
    setPredictions([]);
    setFeedback({});
    setFiles([]);
  }

  return (
      <>
    <form >
      <h1>Production environment</h1>
      <input type="file" onClick={handleClick} onChange={handleOnChange}  />
    </form>
        <table>
          <tbody>
        {files.map((file, index) =>{
          return (<tr>
            <td> <img  src={file} alt="image uploaded"  style={{maxWidth: "400px"}}/></td>
            <td>{predictions && predictions.length > 0 ? <p className="prediction"> It is a {predictions[index]}
            </p> : <p className="prediction"> Loading ... </p>}
              {predictions && predictions.length && <Feedback feedback={feedback} handleFeedback={handleFeedback} index={index}  />}
            </td>
          </tr>)
        })}
          </tbody>
      </table>

        </>

  );
}

export default App;


