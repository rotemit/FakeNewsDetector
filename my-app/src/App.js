import './App.scss';
import {ScanPost} from './Components/ScanPost'
import {Login} from './Components/Login'
import { About } from './Components/About';
import { Contact } from './Components/Contact';
import React, { useState, useEffect } from "react";
import background from "./image.jpeg";

const Link = ({ className, href, children }) => {
  const onClick = (event) => {
    if (event.metaKey || event.ctrlKey) {
      return;
    }

    event.preventDefault();
    window.history.pushState({}, '', href);

    const navEvent = new PopStateEvent('popstate');
    window.dispatchEvent(navEvent);
  };

  return (
    <a onClick={onClick} className={className} href={href}>
      {children}
    </a>
  );
};


const Header = () => {
  return (
    <div className="ui menu">
      <Link href="/" className="item">
        Login
      </Link>
      <Link href="/ScanPost" className="item">
        Scan
      </Link>
      <Link href="/About" className="item">
        About
      </Link>
      <Link href="/Contact" className="item">
        Contact
      </Link>
    </div>
  );
};

const Route = ({ path, children }) => {
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  useEffect(() => {
    const onLocationChange = () => {
      setCurrentPath(window.location.pathname);
    };

    window.addEventListener('popstate', onLocationChange);

    return () => {
      window.removeEventListener('popstate', onLocationChange);
    };
  }, []);

  return currentPath === path ? children : null;
};

const App = () => {
  return (
    <div style={{ backgroundImage: `url(${background})` }}>
      <Header />
      <Route path="/Contact">
        <Contact />
      </Route>
      <Route path="/About">
        <About />
      </Route>
      <Route path="/ScanPost">
        <ScanPost  />
      </Route>
      <Route path="/">
        <Login  />
      </Route>
  </div>
  )
}


// export default withRouter(App);
export default App;
