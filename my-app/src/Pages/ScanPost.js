import "./ScanPost.css";
import React, { useState } from 'react';

export const ScanPost = () => {
  const [text, setText] = useState('');
  const [numOfPosts, setNumOfPosts] = useState(20);
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
      setResult({...Result, semantic: null});  
      setNumOfPosts(20);
      setText('');
      setIsClicked(true);
      const response = await fetch('/scanPost', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: text, numOfPosts: numOfPosts })
      })
      if (response.ok) {
          response.json().then((res) => {
            setResult({trust: res.utv_result, machine: res.machineLearning_result, semantic: res.sentimentAnalyzer_result })
          })
      } 
  }
  let total;
  if (result.trust === -1) {
    total = (Number(result.machine) + Number(result.semantic)) /2 *100;
  } else if ((result.machine === -1)) {
    total = (Number(result.trust) + Number(result.semantic)) /2 *100;
  } else {
     total = (Number(result.trust) + Number(result.machine) + Number(result.semantic)) / 3 * 100;
  }
  const tot = total.toFixed(2);

  return (
    <div className='screen'>
      <div className='form'>
        <div>

        <div className='inputRaw'>
          <input className='urlInput' value={text} placeholder="Post URL"  onChange={e => setText(e.target.value)} />   
          <button class="ui button" type="submit" onClick={handleSubmit}>Submit</button>
        </div>
        <div className='inputRaw'>
          <label>How many posts do you wand to scan?</label>
          <input className='urlInput' value={numOfPosts} placeholder="20"  onChange={e => setNumOfPosts(e.target.value)} />   
        </div>
        {isClicked ? 
          ((result.semantic) ? 
          (<>
            <Result />
            <h1>Total: {tot}</h1>
          </>) : 
          (<div className='loader' />)) : null
        }
        </div>
          
      </div>
    </div>
  )
}