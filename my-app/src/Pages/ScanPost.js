import "./ScanPost.css";
import React, { useState } from 'react';

export const ScanPost = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState({trust:null, machine:null, semantic:null});
  const [isClicked, setIsClicked] = useState(false);

  function Result() {
    return (
      
           <table className='table'>
            <tr>
              <th >Category</th> 
              <th >Value</th>
            </tr>
            <tr>
              <td >Trust result</td> 
              <td >{(result.trust >= 0 ) ? result.trust  : 'N/A'}</td>
            </tr>
            <tr>
              <td >Machine result</td> 
              <td >{(result.machine >= 0 ) ? result.machine  : 'N/A'}</td>
            </tr>
            <tr>
              <td >Semantic result</td> 
              <td >{result.semantic}</td>
            </tr>
           </table>
      
    )
  }

  async function handleSubmit()  {   
      const response = await fetch('/scanPost', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(text)
      })
      if (response.ok) {
          response.json().then((res) => {
            setResult({trust: res.utv_result, machine: res.machineLearning_result, semantic: res.sentimentAnalyzer_result })
          })
      } 
  }
  
    const tot1 = (Number(result.trust) + Number(result.machine) + Number(result.semantic)) / 3 * 100;
    const tot = tot1.toFixed(2);

    return (
      <div className='screen'>
        <div className='form'>
          <div>

          <div className='inputRaw'>
            <input className='urlInput' value={text} placeholder="Post URL"  onChange={e => setText(e.target.value)} />   
            <button class="ui button" type="submit" onClick={handleSubmit}>Submit</button>
          </div>

          {(result.semantic)? 
            <>
            <Result />
            <h2>Total: {tot} </h2>
            </>
           : <p>No result</p>}
          </div>
            
        </div>
      </div>
    )
}