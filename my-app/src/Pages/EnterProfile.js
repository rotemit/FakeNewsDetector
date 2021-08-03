import React, { useState, useEffect } from 'react';
import { Posts } from '../Components/Posts/Posts';
// import { useHistory } from "react-router-dom";
// import {Posts} from "./../Components/Posts/Posts";

export const EnterProfile = () => {
  const [name, setName] = useState('');
  const [posts, setPost] = useState('');

 useEffect(() => {
   fetch('/profile').then(response =>  
    response.json().then(data => {
     setPost(data.name);
     console.log(data.name);
   })
   );
  //  setPost(response);
 }, []);

  return(
    <div className="wrapper">
      <h1>Scan profile</h1>
      <form>
        <fieldset>
          <label>
            <p>Enter url</p>
            <input value={name} onChange={e => setName(e.target.value)} />
          </label>
        </fieldset>
        <button type="submit" onClick={async () => {
          
          const response = await fetch('/profile', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(name)
          });
          if (response.ok) {
            console.log('response worked!')
          }
        }}
        >Submit</button>
        <div>
        <Posts posts={posts}/>
        </div>
        
      </form>
    </div>
    
  )
 
 
}