package com.example.jeanduong.framedplateapplication;


public class CustomRegexToolbox
{
    private String symbole_prefix = "sy\\s*[:\\.]?";
    private String getSymbole_base = "[0-9]+";

    //private String serial_prefix = "s[eé]r(ie)?\\s*[:\\.]?";
    private String serial_prefix = "s[eé]r\\s*[:\\.]?";
    private String serial_base = "[A-Z]+";

    private String manufacturer_prefix = "n[o°]?\\s*[:\\.]?";
    private String getManufacturer_base = "[0-9]+";

    private String warranty_prefix = "gar(antie)?\\s*[:\\.]?";
    private String getWarranty_base = "[0-9]{1,2}(\\s*ans?)?";

    private String commission_prefix = "date";
    private String getCommission_base = "(0?[1-9])|(1[0-2])[/\\-]([0-9]{2})|([0-9]{4})";

    public CustomRegexToolbox(){super();};

    public String getSymbole_prefix() {
        return symbole_prefix;
    }

    public String getGetSymbole_base() {
        return getSymbole_base;
    }

    public String getSerial_prefix() {
        return serial_prefix;
    }

    public String getSerial_base() {
        return serial_base;
    }

    public String getManufacturer_prefix() {
        return manufacturer_prefix;
    }

    public String getGetManufacturer_base() {
        return getManufacturer_base;
    }

    public String getWarranty_prefix() {
        return warranty_prefix;
    }

    public String getGetWarranty_base() {
        return getWarranty_base;
    }

    public String getCommission_prefix() {
        return commission_prefix;
    }

    public String getGetCommission_base() {
        return getCommission_base;
    }
}
