import "./ScanPost.scss";
import React, { useState } from 'react';
import { Loader, Form, SubmitButton, Input } from "./BasicComponents";

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
              <td >{(result.trust >= 0 ) ? (Number(result.trust) * 100).toFixed(2) + '%' : 'N/A'}</td>
            </tr>
            <tr>
              <td >Machine result</td> 
              <td >{(result.machine >= 0 ) ? (Number(result.machine) * 100).toFixed(2) + '%'  : 'N/A'}</td>
            </tr>
            <tr>
              <td >Sentiment result</td> 
              <td >{(result.semantic >= 0 ) ? (Number(result.semantic) * 100).toFixed(2) + '%'  : 'N/A'}</td>
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
  let total = 0;
  let devide = 0;
  if (result.trust !== -1) {
    total += Number(result.trust);
    devide++;
  } 
  if (result.machine !== -1) {
    total += Number(result.machine);
    devide++;
  } 
  if (result.semantic !== -1) {
    total += Number(result.semantic);
    devide++;
  }
  const tot = (devide === 0 ? 'N/A' : (total / devide * 100).toFixed(2));

  return (
    <div className='screen'>
      <Form>
        <div className='fields'>  
          <Input label='Enter URL/text' value={text} onChange={e => setText(e.target.value)} />
          <Input label='Number of posts' value={numOfPosts} onChange={e => setNumOfPosts(e.target.value)} />
          <SubmitButton onSubmit={handleSubmit} />
        </div>
  
        {isClicked ? 
          ((result.semantic) ? 
          (<>
            <Result />
            <div className='total'>Total: {tot}%</div>
          </>) : 
          (<Loader />)) : null
        }
          
      </Form>
    </div>
  )
}