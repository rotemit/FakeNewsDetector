import React from 'react';
// import { Form } from 'semantic-ui-react';
import Feature from '../Components/Features/Feature';
import './HomePage.css'
export const HomePage = () => {
    // const {todo, setTodo} = useState([])
   
    return(
        <div className="header">
            <label className='title'>Welcome to detect fake news app!</label>
            {/* <div className='.form-features'> */}
                <div className='features'>
                    <Feature text="Scan Post" className='feature' />
                    <Feature text="Scan Post" />
                {/* </div>             */}
                {/* <div className='features-end'> */}
                    <Feature text="Scan Post" />
                    <Feature text="Scan Post" />
                </div>
            {/* </div> */}

            
        </div>
       
        
    )
}

